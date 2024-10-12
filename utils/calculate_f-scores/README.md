# Calculate F-Scores

Scripts for evaluating matching results for ROR ID predictions.

- `calculate_f_score_best_match.py`: Evaluates based on the highest scoring prediction.
- `calculate_f_score_openalex.py`: Evaluates considering all predictions for each input.

## Usage

```
python calculate_f_score_best_match.py -i <input_csv> [-o <output_csv>]
```

## Input

- Input CSV must contain columns: `ror_id`, `predicted_ror_id`, `prediction_scores`
- For `calculate_f_score_openalex.py`, `predicted_ror_id` and `prediction_scores` can contain multiple values separated by semicolons

## Output

Both scripts calculate and display:
- Precision
- Recall
- F1 Score
- F0.5 Score
- Specificity

Results are printed to console and saved to CSV files.

### Output Files

- Both scripts output:
  - A CSV with classification results for each input row
  - A CSV with overall performance metrics

## Evaluation Method

- `calculate_f_score_best_match.py`: Considers only the highest scoring prediction for each input.
- `calculate_f_score_openalex.py`: Considers all predictions for each input, treating each prediction as a separate case in a confusion matrix calculation.

