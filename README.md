# OpenAlex ROR Predictor
Modified version of [jpbarrett13](https://github.com/jpbarrett13)'s [OpenAlex institution ID prediction service](https://github.com/ourresearch/openalex-institution-parsing) that predicts both OpenAlex institution IDs and ROR IDs from affiliation strings.

## Overview
This code uses the v2 version of OpenAlex's institutional classification models to predict institution IDs and then map them their corresponding ROR IDs. For more details on the original work, see OpenAlex's [paper](https://docs.google.com/document/d/1ppbKRVtyneWc7Hjpo8TOm57YLGx1C2Oo/) and [jpbarrett13](https://github.com/jpbarrett13)'s [notebooks](https://github.com/ourresearch/openalex-institution-parsing/tree/main/V2) for model training and inference.

## Installation
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the OpenAlex v2 model artifacts as described in the [notes](https://github.com/ourresearch/openalex-institution-parsing/tree/main/V2) section of the OpenAlex institution parsing repo.
   ```
   aws s3 cp s3://openalex-institution-tagger-model-artifacts/ . --recursive
   ```

3. Run the `download_and_parse_openalex_institutions.py` script to create the OpenAlex institution ROR ID mapping pkl file and place this inside the `institution_tagger_v2_artifacts` directory. This requires you have the AWS command line tool installed. Additional details are provided in the [repo for the download script](https://github.com/adambuttrick/openalex-ror-predictor/tree/main/utils/download_and_parse_openalex_institutions).

## Usage
There are two main scripts:

1. `openalex_ror_predictor.py`: A command-line tool for batch prediction of affiliation strings.
2. `institution_tagger.py`: The core class `InstitutionTagger` that handles the prediction logic.

### Command-line tool usage
To use the command-line tool:

```
python openalex_ror_predictor.py -i <input_file.csv> -o <output_file.csv>
```

The input CSV file should contain an 'affiliation_string' column with the affiliation strings to be processed.

### Python Usage
To use the `InstitutionTagger` class in your code:

```python
from institution_tagger import InstitutionTagger

tagger = InstitutionTagger(model_path="institution_tagger_v2_artifacts")
affiliation_strings = ['University of Michigan, Ann Arbor, USA', 'Getty Conservation Institute, Los Angeles']
predictions = tagger.predict(affiliation_strings)
```

## Output
The predictor returns a list of dictionaries, where each dictionary corresponds to an input affiliation string and contains the following keys:

- `affiliation_string`: The original input affiliation string.
- `institution_id`: A list of predicted OpenAlex institution IDs.
- `ror_id`: A list of corresponding ROR IDs (where available).
- `score`: A list of prediction scores corresponding to each institution ID.
- `category`: A list of prediction categories corresponding to each institution ID.

Example output:

```python
[
    {
        'affiliation_string': 'University of Michigan, Ann Arbor, USA; Getty Conservation Institute, Los Angeles',
        'institution_id': [27837315, 200193707],
        'ror_id': ['https://ror.org/00jmfr291', 'https://ror.org/019496w77'],
        'score': [0.95, 0.85],
        'category': ['model_match', 'string_match']
    },
    {
        'affiliation_string': 'Getty Conservation Institute, Los Angeles',
        'institution_id': [200193707],
        'ror_id': ['https://ror.org/019496w77'],
        'score': [0.90],
        'category': ['basic_thresh']
    }
]
```

