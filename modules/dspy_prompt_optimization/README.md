# DSPy Prompt Optimization

A comprehensive toolkit for developing, optimizing, and evaluating prompts using [DSPy](https://github.com/allenai/dspy). This project streamlines prompt engineering workflows through Python scripts and Poetry dependency management.

## Features
- **build_optimizer.py**: Robust command-line tool that evaluates unoptimized ChainOfThought prompts and generates optimized versions using MIPROv2 algorithm
- **prompt_optimizer.py**: Production-ready script for refining prompts using pre-trained optimizers with contextual awareness
- **generate_dspy_data.py**: Automated synthetic training data generation through language model interactions

## Installation
1. Install Poetry package manager:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
2. Set up project dependencies:
    ```bash
    poetry install
    ```

## Usage
### Build an Optimizer
Create an optimized prompt model:
```bash
poetry run build_optimizer --data_path data_for_dspy.json --output_path my_optimizer.json
```

### Optimize Prompts
Refine prompts with trained optimizer:
```bash
poetry run prompt_optimizer --prompt '{"about_me":"...", "context":"...", "question":"..."}'
```

### Generate Training Data
Create synthetic datasets:
```bash
poetry run generate_data --data_path ./data --output_path ./output.json
```

## Project Structure
- **dspy_prompt_optimization/**: Core implementation files
  - Prompt generation utilities
  - Optimization algorithms
  - Evaluation frameworks
- **pyproject.toml**: Poetry configuration and dependency specifications
- **README.md**: Documentation and setup guide
