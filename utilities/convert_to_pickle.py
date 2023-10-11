import argparse
import csv
import pickle
import re

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        return list(csv_reader)


def transform_data(rows):
    transformed_data = {}
    for row in rows:
        id_url, ror_url = row
        numerical_id = re.search(r'I(\d+)', id_url)
        if numerical_id:
            transformed_data[numerical_id.group(1)] = ror_url
    return transformed_data


def write_pickle(data):
    with open("institution_ror_id_mapping.pkl", "wb") as file:
        pickle.dump(data, file)


def parse_args():
    parser = argparse.ArgumentParser(description="Transform CSV to Pickle")
    parser.add_argument("-i", "--input_file", type=str,
                        help="Path to the input CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv(args.input_file)
    data_dict = transform_data(rows)
    write_pickle(data_dict)


if __name__ == "__main__":
    main()
