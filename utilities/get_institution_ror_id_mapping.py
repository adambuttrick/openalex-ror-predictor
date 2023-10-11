import requests
import csv
import json

API_URL = "https://api.openalex.org/institutions?per-page=200&cursor={}"
OUTPUT_CSV = "institution_ror_id_mapping.csv"
MAX_RETRIES = 3


def fetch_data_from_api(cursor="*"):
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(API_URL.format(cursor))
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            next_cursor = data.get("meta", {}).get("next_cursor", None)
            extracted_data = [(result.get("id", None), result.get(
                "ror", None)) for result in results]
            return extracted_data, next_cursor
        except requests.RequestException as e:
            print(f"Error fetching data from API: {e}")
            continue
    print(f"Failed to fetch data after {MAX_RETRIES} retries.")
    return [], None


def write_to_csv(data):
    with open(OUTPUT_CSV, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)


def main():
    write_to_csv(["ID", "ROR"])
    cursor = "*"
    while cursor:
        data, cursor = fetch_data_from_api(cursor)
        if data:
            write_to_csv(data)
        else:
            break


if __name__ == "__main__":
    main()
