import csv
import argparse
import logging
from institution_tagger import InstitutionTagger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Institution Tagger Prediction Script')
    parser.add_argument('-i', '--input_file', required=True,
                        help='Path to the input CSV file')
    parser.add_argument('-o', '--output_file', required=True,
                        help='Path to the output CSV file')
    return parser.parse_args()


def read_input_data(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            if 'affiliation_string' not in reader.fieldnames:
                raise ValueError(
                    "The input file must contain an 'affiliation_string' column.")

            data = []
            null_count = 0
            for row in reader:
                if not row['affiliation_string']:
                    null_count += 1
                    row['affiliation_string'] = ''
                data.append(row)

            if null_count > 0:
                logger.warning(f"Found {null_count} empty values in affiliation column. Replaced with empty strings.")

            return data
    except FileNotFoundError:
        logger.error(f"Error: The file {input_file} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error reading the input file: {e}")
        raise


def run_inference(data):
    tagger = InstitutionTagger(model_path="institution_tagger_v2_artifacts")
    affiliation_strings = [row['affiliation_string'] for row in data]
    predictions = tagger.predict(affiliation_strings)

    for row, prediction in zip(data, predictions):
        row['predicted_institution_id'] = prediction['institution_id']
        row['predicted_ror_id'] = prediction['ror_id']
        row['prediction_scores'] = prediction['score']
        row['prediction_categories'] = prediction['category']

    return data


def clean_ror_id(value):
    if value == 'None':
        return ''
    else:
        return ';'.join(filter(lambda x: x != 'None', value.split(';')))


def write_output_data(data, output_file):
    try:
        columns_to_write = ['ror_id', 'affiliation_string', 'predicted_institution_id',
                            'predicted_ror_id', 'prediction_scores', 'prediction_categories']

        with open(output_file, 'w',encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=columns_to_write)
            writer.writeheader()

            for row in data:
                row['predicted_ror_id'] = clean_ror_id(row['predicted_ror_id'])
                writer.writerow({k: row[k] for k in columns_to_write})

        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error writing to the output file: {e}")
        print(f"Available columns: {', '.join(data[0].keys())}")


def main():
    args = parse_args()
    data = read_input_data(args.input_file)
    data_with_predictions = run_inference(data)
    write_output_data(data_with_predictions, args.output_file)


if __name__ == '__main__':
    main()
