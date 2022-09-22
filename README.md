# openalex-ror-predictor
Minor tweaking of @jpbarrett13's [OpenAlex institution ID prediction service](https://github.com/ourresearch/openalex-institution-parsing/tree/main/V1/003_Deploy/model_to_api) to spit out ROR IDs.

#overview
The team at OpenAlex trained a text classification model on their affiliation strings to predict institution IDs. By adding a mapping between OpenAlex institution IDs and ROR IDs, the same model can be used to predict ROR IDs. See OpenAlex's [paper](https://docs.google.com/document/d/1ppbKRVtyneWc7Hjpo8TOm57YLGx1C2Oo/) and @jpbarrett13's [notebooks](https://github.com/ourresearch/openalex-institution-parsing/tree/main/V1) for full details.

# installation
* pip install -r requirements.txt
* Download the model artifacts as described in the [notes](https://github.com/ourresearch/openalex-institution-parsing/tree/main/V1) section of the OpenAlex repository.
* Add institution_ror_id_mapping.pkl to the model artifacts directory.
* Add model artifacts file path to the predictor class instantiation in app.py

# usage
Start up the flask app and post the affiliation string to the invocations route. See test.py for an example. Where both an institution ID and a ROR ID exist, both values will be returned. Where no ROR ID exists for an institution ID, only the institution ID will be returned.

# limitations
Please review all of @jpbarrett13's [notebooks](https://github.com/ourresearch/openalex-institution-parsing/tree/main/V1) to understand how the model was trained. Prediction success is determined (in part) by the amount of training affiliation data that was available for any given institution or ROR ID. In addition, because this is a classification model, affiliation strings for institutions that do not exist in OpenAlex will not return correct results. Likewise, ROR IDs cannot be returned for the affiliation strings from 465 ROR IDs which lack corresponding institution IDs in OpenAlex. 

