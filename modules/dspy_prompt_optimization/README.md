# DSPy Prompt Optimization

A Python package for optimizing Large Language Model (LLM) prompts using DSPy framework. This tool helps developers automatically fine-tune and improve their prompting strategies for better and more consistent LLM responses.

## Description

DSPy Prompt Optimization provides a systematic approach to optimize prompts for Large Language Models. It leverages the DSPy framework to:
- Automatically discover effective prompting patterns
- Improve prompt reliability and consistency
- Reduce manual prompt engineering effort
- Enable data-driven prompt optimization

## Installation

Install the package using pip:

```bash
pip install dspy-prompt-optimization
```

Or using Poetry:

```bash
poetry add dspy-prompt-optimization
```

## Basic Usage

```python
from dspy_prompt_optimization import PromptOptimizer

# Initialize the optimizer
optimizer = PromptOptimizer()

# Define your initial prompt
initial_prompt = "Summarize the following text:"

# Optimize the prompt using your training data
optimized_prompt = optimizer.optimize(
    initial_prompt=initial_prompt,
    training_data=your_training_data
)

# Use the optimized prompt
results = optimizer.generate(optimized_prompt, input_text)
```

For more detailed examples and advanced usage, please refer to the documentation.

