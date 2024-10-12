# Calculate F-Scores

Script for evaluating matching results for the OpenAlex predictor. 

- `calculate_f_score_best_match.py` Evaluates based on highest scoring prediction.

## Usage

```
python calculate_f_score_best_match.py -i <input_csv> [-o <output_csv>]
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