# OpenAlex Institution-ROR ID Mapping

Downloads, extracts, and processes OpenAlex institution data, creating a CSV file and a pickle file mapping OpenAlex IDs to ROR IDs.

## Requirements
- AWS CLI

## Installation

1. Install the AWS CLI tool
3. Install required Python packages:
   ```
   pip install tqdm
   ```

## Usage

```
python download_and_parse_openalex_institutions.py [-h] [-o OUTPUT] [-p PICKLE] [-l LOG] [-d DOWNLOAD_DIR]
```

## Arguments

- `-o, --output`: Output CSV file path (default: output.csv)
- `-p, --pickle`: Output pickle file path (default: institutions.pkl)
- `-l, --log`: Log file path (default: institution_processing.log)
- `-d, --download_dir`: Directory to download and process files (default: openalex-institutions)

## Process

1. Downloads OpenAlex institution files from S3 using AWS CLI
2. Processes gzipped JSONL files
3. Extracts OpenAlex IDs and ROR IDs
4. Outputs results to CSV and pickle files

## Output

- CSV file with OpenAlex ID and ROR ID pairs
- Pickle file with a dictionary mapping OpenAlex IDs to ROR IDs