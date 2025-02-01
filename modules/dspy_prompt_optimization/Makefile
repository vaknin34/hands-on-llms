.PHONY: Deals with dspy prompt optimization related tasks.

# install Poetry and Python dependencies
install:
	@echo "Installing python environment..."
	poetry install

# generate data for dspy optimation based on the training pipeline dataset and more syntethic data.
generate-data:
	@echo "Generating data for DSPY optimation..."
	poetry run generate_data --data_path ../training_pipeline/dataset --output_path dataset/data_for_dspy.json
	
# build the miprov2 optimizer program.
build-optimizer:
	@echo "Building DSPY prompt optimizer..."
	poetry run build_optimizer --data_path dataset/data_for_dspy.json --output_path optimizer/mipro_optimizer_v2.json
