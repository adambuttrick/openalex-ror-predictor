# Calculate F-Scores

Evaluates matching results for ROR ID with multiple inputs and outputs using a confusion matrix.

## Usage

```
python calculate_f_score_confusion_matrix.py -i <input_csv> [-o <output_csv>] [-m <metrics_csv>]
```

## Arguments

- `-i`, `--input`: Input CSV file (required)
- `-o`, `--output`: Output CSV file with classification results (optional, default: `<input_filename>_classified.csv`)
- `-m`, `--metrics`: Output CSV file for metrics (optional, default: `<input_filename>_metrics.csv`)

## Input

The input CSV must contain the following columns:
- `ror_id`: The true ROR ID
- `predicted_ror_id`: Predicted ROR ID(s), separated by semicolons if multiple
- `prediction_scores`: Corresponding prediction scores, separated by semicolons if multiple

## Output

The script calculates and displays:
- Precision
- Recall
- F1 Score
- F0.5 Score
- Specificity

Results are printed to the console and saved to CSV files.

### Output Files

1. Classification results CSV: Contains the original input data with additional columns for TP, FP, TN, and FN.
2. Metrics CSV: Contains overall performance metrics (Precision, Recall, F1 Score, F0.5 Score, Specificity).

## Evaluation Method

The script considers all predictions for each input, treating each prediction as a separate case in a confusion matrix calculation. It handles cases where there may be multiple predictions for a single true ROR ID, as well as cases with no true ROR ID or no predictions.

