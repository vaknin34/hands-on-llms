import logging
import json
import dspy
import fire

from tools.dspy_financial_bot import DspyFinancialBotLM

logger = logging.getLogger(__name__)


def run_local():

    dspy.configure(lm=DspyFinancialBotLM())

    # Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
    qa = dspy.ChainOfThought('question -> answer')

    # Run with the default LM configured with `dspy.configure` above.
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print(response.answer)



if __name__ == "__main__":
    fire.Fire(run_local)
