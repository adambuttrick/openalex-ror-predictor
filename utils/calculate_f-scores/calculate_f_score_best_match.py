import csv
import sys
import os
import argparse


def load_results_set(f):
    with open(f, 'r+', encoding='utf-8-sig') as f_in:
        reader = csv.DictReader(f_in)
        results_set = [row for row in reader]
        for row in results_set:
            row['match'] = calculate_match(row)
    return results_set


def calculate_match(row):
    ror_id = row['ror_id'].strip() if 'ror_id' in row else ''
    predicted_ror_ids = row['predicted_ror_id'].split(';')
    prediction_scores = row['prediction_scores'].split(';')
    valid_predictions = [(pred_ror_id.strip(), float(score))
                         for pred_ror_id, score in zip(predicted_ror_ids, prediction_scores)
                         if pred_ror_id.strip() and score.strip()]
    if not ror_id:
        if not valid_predictions:
            return 'TN'
        else:
            return 'FP'
    else:
        if not valid_predictions:
            return 'FN'
        highest_scoring_ror_id, highest_score = max(valid_predictions, key=lambda x: x[1])
        if ror_id == highest_scoring_ror_id:
            return 'TP'
        else:
            return 'FP'


def calculate_counts(results_set):
    true_pos = sum(1 for row in results_set if row['match'] == 'TP')
    false_pos = sum(1 for row in results_set if row['match'] == 'FP')
    false_neg = sum(1 for row in results_set if row['match'] == 'FN' or (
        row['match'] == 'NP' and row['ror_id']))
    true_neg = sum(1 for row in results_set if row['match'] == 'TN')
    print(true_pos, false_pos, false_neg, true_neg)
    return true_pos, false_pos, false_neg, true_neg


def safe_div(n, d, default_ret=0):
    return n / d if d != 0 else default_ret


def calculate_metrics(true_pos, false_pos, false_neg, true_neg):
    precision = safe_div(true_pos, true_pos + false_pos)
    recall = safe_div(true_pos, true_pos + false_neg)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    beta = 0.5
    f0_5_score = safe_div((1 + beta**2) * (precision * recall),
                          (beta**2 * precision) + recall)
    specificity = safe_div(true_neg, true_neg + false_pos)
    return precision, recall, f1_score, f0_5_score, specificity


def write_to_csv(filename, precision, recall, f1_score, f0_5_score, specificity):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Precision", "Recall", "F1 Score",
                         "F0.5 Score", "Specificity"])
        writer.writerow([precision, recall, f1_score, f0_5_score, specificity])


def write_results_with_classification(input_file, output_file, results_set):
    with open(input_file, 'r', encoding='utf-8-sig') as f_in, open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['classification']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row, result in zip(reader, results_set):
            row['classification'] = result['match']
            writer.writerow(row)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Calculate f-scores for a given CSV file and add classification column.')
    parser.add_argument('-i', '--input', help='Input CSV file', required=True)
    parser.add_argument('-o', '--output', help='Output CSV file', default=None)
    parser.add_argument('-m', '--metrics',
                        help='Metrics output CSV file', default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = f'{os.path.splitext(args.input)[0]}_classified.csv'
    if args.metrics is None:
        args.metrics = f'{os.path.splitext(args.input)[0]}_metrics.csv'
    return args


def main():
    args = parse_arguments()
    results_set = load_results_set(args.input)
    write_results_with_classification(args.input, args.output, results_set)
    true_pos, false_pos, false_neg, true_neg = calculate_counts(results_set)
    precision, recall, f1_score, f0_5_score, specificity = calculate_metrics(
        true_pos, false_pos, false_neg, true_neg)
    print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}\nF0.5 Score: {f0_5_score}\nSpecificity: {specificity}")
    write_to_csv(args.metrics, precision, recall,
                 f1_score, f0_5_score, specificity)


if __name__ == "__main__":
    main()
