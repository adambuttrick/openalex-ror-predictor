# Calculate F-Scores

Scripts for evaluating matching results for the OpenAlex predictor:

1. `calculate_f_score_best_match.py`: Evaluates based on highest scoring prediction.
2. `calculate_f_score_label_in_prediction.py`: Evaluates based on presence of correct ROR ID in predictions.

## Distinction Between Scripts

- `calculate_f_score_best_match.py`:
  - Considers only the highest-scoring predicted ROR ID.
  - Match is correct only if the highest-scoring prediction matches the dataset ROR ID label.
  - More stringent evaluation.

- `calculate_f_score_label_in_prediction.py`:
  - Checks if the correct ROR ID is present anywhere in the list of predictions.
  - Match is correct if the dataset ROR ID label is found in the prediction set, regardless of score or number of prediction.
  - More lenient evaluation.

## Usage

```
python <script_name>.py -i <input_csv> [-o <output_csv>]
```

- Input CSV must contain columns: `ror_id`, `predicted_ror_id`, `prediction_scores`
- Output CSV contains precision, recall, F1 score, F0.5 score, and specificity


## Output

Both scripts calculate and display:
- Precision
- Recall
- F1 Score
- F0.5 Score
- Specificity

Results are printed to console and saved to a CSV file.