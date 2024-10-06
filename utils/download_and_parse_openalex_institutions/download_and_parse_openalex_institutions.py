import os
import re
import csv
import gzip
import json
import pickle
import logging
import argparse
import multiprocessing
import subprocess
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download, extract OpenAlex IDs and ROR ID values from the OpenAlex data snapshot (gzipped JSONL files).")
    parser.add_argument("-o", "--output", default="output.csv",
                        help="Output CSV file path (default: output.csv)")
    parser.add_argument("-p", "--pickle", default="institutions.pkl",
                        help="Output pickle file path (default: institutions.pkl)")
    parser.add_argument("-l", "--log", default="institution_processing.log",
                        help="Log file path (default: institution_processing.log)")
    parser.add_argument("-d", "--download_dir", default="openalex-institutions",
                        help="Directory to download and process files (default: openalex-institutions)")
    return parser.parse_args()


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def download_files(download_dir):
    logging.info("Starting download of OpenAlex institution files")
    os.makedirs(download_dir, exist_ok=True)
    command = [
        "aws", "s3", "sync",
        "s3://openalex/data/institutions/",
        download_dir,
        "--no-sign-request"
    ]
    try:
        subprocess.run(command, check=True)
        logging.info("Download completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downloading files: {e}")
        raise


def discover_files(input_dir):
    gz_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.gz'):
                gz_files.append(os.path.join(root, file))
    return gz_files


def extract_id(openalex_id):
    match = re.search(r'I(\d+)$', openalex_id)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid OpenAlex ID format: {openalex_id}")


def process_jsonl_file(file_path):
    with gzip.open(file_path, 'rt') as f_in:
        for line in f_in:
            try:
                record = json.loads(line)
                openalex_id = extract_id(record['id'])
                ror = record.get('ror', '')
                yield openalex_id, ror
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.warning(f"Error processing line in {file_path}: {e}")


def write_csv(data, output_file):
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['OpenAlex ID', 'ROR'])
        writer.writerows(data)


def write_pickle(data, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(dict(data), f_out)


def process_file_wrapper(file_path):
    try:
        return list(process_jsonl_file(file_path))
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []


def main():
    args = parse_arguments()
    setup_logging(args.log)

    download_files(args.download_dir)

    logging.info("Starting OpenAlex ID and ROR extraction")
    files = discover_files(args.download_dir)
    logging.info(f"Found {len(files)} .gz files to process")

    all_data = []
    with multiprocessing.Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_file_wrapper, files), total=len(files), desc="Processing files"):
            all_data.extend(result)

    write_csv(all_data, args.output)
    logging.info(f"Extraction complete. Results written to {args.output}")

    write_pickle(all_data, args.pickle)
    logging.info(f"Pickle file with OpenAlex ID - ROR ID mapping written to {args.pickle}")


if __name__ == "__main__":
    main()
