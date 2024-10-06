import csv
import argparse
import logging
import pandas as pd
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
    args = parser.parse_args()
    return args


def read_input_data(input_file):
    try:
        df = pd.read_csv(input_file)
        if 'affiliation_string' not in df.columns:
            raise ValueError(
                "The input file must contain an 'affiliation_string' column.")
        nan_count = df['affiliation_string'].isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in affiliation column. Replacing with empty strings.")
        df['affiliation_string'] = df['affiliation_string'].fillna('')
        return df
    except FileNotFoundError:
        logger.error(f"Error: The file {input_file} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error reading the input file: {e}")
        raise


def run_inference(df):
    tagger = InstitutionTagger(model_path="institution_tagger_v2_artifacts")
    predictions_df = tagger.predict(df)
    df['predicted_institution_id'] = predictions_df['institution_id']
    df['predicted_ror_id'] = predictions_df['ror_id']
    df['prediction_scores'] = predictions_df['score']
    df['prediction_categories'] = predictions_df['category']
    return df


def clean_ror_id(value):
    if value == 'None':
        return ''
    else:
        return ';'.join(filter(lambda x: x != 'None', value.split(';')))


def write_output_data(df, output_file):
    try:
        columns_to_write = ['ror_id', 'affiliation_string', 'predicted_institution_id', 'predicted_ror_id', 'prediction_scores', 'prediction_categories']
        df['predicted_ror_id'] = df['predicted_ror_id'].apply(clean_ror_id)
        df[columns_to_write].to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error writing to the output file: {e}")
        print(f"Available columns: {df.columns}")


def main():
    args = parse_args()
    df = read_input_data(args.input_file)
    df_with_predictions = run_inference(df)
    write_output_data(df_with_predictions, args.output_file)


if __name__ == '__main__':
    main()
