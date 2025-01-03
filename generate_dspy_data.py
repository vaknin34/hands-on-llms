import os
import json
from pathlib import Path
import random
import argparse

import dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI


dotenv.load_dotenv()

ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_API_VERSION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training data samples.")
    parser.add_argument('--data_path', type=Path, required=True, help="Path to the directory containing the dataset JSON files.")
    parser.add_argument('--output_path', type=Path, default=Path('data_for_dspy.json'), help="Path to save the generated data JSON file.")
    args = parser.parse_args()
    return args


class AgentRequestResponseExample(BaseModel):
    '''A request and response for an assistant.'''
    about_me: str = Field(description='Information about the writer.')
    context: str = Field(description='The relevant context the response is based on.')
    question: str = Field(description='The request to the assistant.')
    response: str = Field(description='The response the agent returned.')


def get_data(data_dir: Path) -> list[dict]:
    data = []
    for data_file in data_dir.rglob('*.json'):
        with open(data_file) as f:
            data += json.load(f)
    return data


def generate_samples(structured_llm: AzureChatOpenAI, data: list[dict], num_sample: int, k: int=3) -> list[AgentRequestResponseExample]:
        samples = []
        prompt = 'Given the following data examples:\n{}.\nGenerate one more data example with the same structure.'
        for i in range(num_sample):
            print(f'Generating sample {i + 1}\{num_sample}...')
            samples.append(structured_llm.invoke(prompt.format(random.sample(data, k))))
        return samples


if __name__ == '__main__':
    args = parse_args()

    llm = AzureChatOpenAI(
        azure_endpoint=ENDPOINT,
        azure_deployment=DEPLOYMENT,
        openai_api_version=API_VERSION,
        api_key=API_KEY
    )

    data = get_data(args.data_path)

    structured_llm = llm.with_structured_output(AgentRequestResponseExample)

    results = generate_samples(structured_llm, data, 200 - len(data))

    with open(args.output_path, 'w') as f:
        json.dump(data + [r.model_dump() for r in results], f, indent=4)
