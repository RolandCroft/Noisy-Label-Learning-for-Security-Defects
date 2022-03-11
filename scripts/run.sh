#!/bin/bash

# Requirements
pip install requirements.txt

# Extract features
echo "Extracting Code Features"
while IFS=, read -r a; do
	python code/svp/software_metrics.py "${a}"
  python code/svp/text_encoder.py "bert" "${par[0]}"
  python code/svp/text_encoder.py "bow" "${par[0]}"
done < scripts/inputs/inputs.csv

# Construct and Evaluate SVP Models
echo "Evaluating SVP Models"
while IFS=, read -r a b c d; do
	python code/svp/main.py -r "${a}" -m "${b}" -f "${c}" -t "${d}"
done < scripts/inputs/ml_inputs.csv

# Get additional results for transductive learning
echo "Getting Results for Transductive Learning"
python code/svp/transductive_learning.py
