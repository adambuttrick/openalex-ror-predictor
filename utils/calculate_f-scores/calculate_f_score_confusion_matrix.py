import csv
import sys
import os
import argparse
from collections import defaultdict


def process_ror_ids(ror_id_string):
    return set(id.strip() for id in ror_id_string.split(';') if id.strip())


def load_results_set(f):
    with open(f, 'r+', encoding='utf-8-sig') as f_in:
        reader = csv.DictReader(f_in)
        results_set = []
        for row in reader:
            processed_row = {
                'ror_id': process_ror_ids(row['ror_id']),
                'predicted_ror_ids': process_ror_ids(row['predicted_ror_id'])
            }
            processed_row['confusion_matrix'] = calculate_confusion_matrix(
                processed_row)
            results_set.append(processed_row)
    return results_set


def calculate_confusion_matrix(row):
    true_ids = row['ror_id']
    pred_ids = row['predicted_ror_ids']
    TP, FP, TN, FN = 0, 0, 0, 0
    if len(true_ids) == 0:
        if len(pred_ids) == 0:
            TN = 1
        else:
            FP = len(pred_ids)
    elif len(pred_ids) == 0:
        FN = len(true_ids)
    else:
        TP = sum(1 for x in pred_ids if x in true_ids)
        FP = sum(1 for x in pred_ids if x not in true_ids)
        FN = sum(1 for x in true_ids if x not in pred_ids)
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}


def calculate_counts(results_set):
    total_counts = defaultdict(int)
    for row in results_set:
        for key, value in row['confusion_matrix'].items():
            total_counts[key] += value
    return total_counts


def safe_div(n, d, default_ret=0):
    return n / d if d != 0 else default_ret


def calculate_metrics(counts):
    true_pos = counts['TP']
    false_pos = counts['FP']
    false_neg = counts['FN']
    true_neg = counts['TN']
    precision = safe_div(true_pos, true_pos + false_pos)
    recall = safe_div(true_pos, true_pos + false_neg)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    beta = 0.5
    f0_5_score = safe_div((1 + beta**2) * (precision * recall),
                          (beta**2 * precision) + recall)
    specificity = safe_div(true_neg, true_neg + false_pos)
    return precision, recall, f1_score, f0_5_score, specificity


def write_to_csv(filename, metrics):
    with open(filename, 'w') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Precision", "Recall", "F1 Score",
                         "F0.5 Score", "Specificity"])
        writer.writerow(metrics)


def write_results_with_classification(input_file, output_file, results_set):
    with open(input_file, 'r', encoding='utf-8-sig') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['TP', 'FP', 'TN', 'FN']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row, result in zip(reader, results_set):
            row.update(result['confusion_matrix'])
            row['ror_id'] = ';'.join(sorted(result['ror_id']))
            row['predicted_ror_id'] = ';'.join(
                sorted(result['predicted_ror_ids']))
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

    counts = calculate_counts(results_set)
    metrics = calculate_metrics(counts)

    write_to_csv(args.metrics, metrics)

    print(f"Precision: {metrics[0]}")
    print(f"Recall: {metrics[1]}")
    print(f"F1 Score: {metrics[2]}")
    print(f"F0.5 Score: {metrics[3]}")
    print(f"Specificity: {metrics[4]}")


if __name__ == "__main__":
    main()
