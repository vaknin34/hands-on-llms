import logging
import json

import fire


from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)


def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        #context_entity_recall,
        context_recall,
        #context_relevancy,
        #context_utilization,
        faithfulness
    )
    from ragas.metrics.context_precision import context_relevancy
    metrics = [
        #context_utilization,
        context_relevancy,
        context_recall,
        answer_similarity,
        #context_entity_recall,
        #answer_correctness,
        faithfulness
    ]

    with open(testset_path, "r") as f:
        data = json.load(f)
        aggregate_score = {
            'ragas_score': 0.0000,
            'context_relevancy': 0.0000,
            'context_recall': 0.0000,
            'answer_similarity': 0.0000,
            'faithfulness': 0.0000
        }
        start = 1
        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)
            ragas_score = evaluate_w_ragas(query=elem["question"], context=output_context.split('\n'), output=response, ground_truth=elem["response"], metrics=metrics)
            
            # Update the aggregate score
            for key in aggregate_score:
                aggregate_score[key] += ragas_score[key]

            # Log the score
            logger.info("Score=%s", ragas_score)
            # Log the iteration number
            logger.info(f"Iteration={start} of {len(data)}")
            start += 1
        
        # Calculate the average score
        for key in aggregate_score:
            aggregate_score[key] /= len(data)

    return aggregate_score


if __name__ == "__main__":
    fire.Fire(run_local)
