#!/usr/bin/env python3

import json
from pathlib import Path
import random
import argparse
import dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training data samples.")
    parser.add_argument('--data_path', type=Path, required=True, help="Path to the directory containing the dataset JSON files.")
    parser.add_argument('--output_path', type=Path, help="Path to save the generated data JSON file.")
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


def generate_samples(structured_llm: ChatOpenAI, data: list[dict], num_sample: int, k: int=3) -> list[AgentRequestResponseExample]:
        samples = []
        prompt = 'Given the following data examples:\n{}.\nGenerate one more data example with the same structure.'
        for i in range(num_sample):
            print(f'Generating sample {i + 1}\{num_sample}...')
            samples.append(structured_llm.invoke(prompt.format(random.sample(data, k))))
        return samples

def main():
    args = parse_args()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        max_retries=3
    )

    data = get_data(args.data_path)

    structured_llm = llm.with_structured_output(AgentRequestResponseExample)

    results = generate_samples(structured_llm, data, 200 - len(data))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(data + [r.model_dump() for r in results], f, indent=4)


if __name__ == '__main__':
    main()