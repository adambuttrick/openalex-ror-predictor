import flask
import json
import pandas as pd
from predictor import Predictor


# Create class instance with path to the model files downloaded from OpenAlex.
# See "NOTES" section here for instructions on downloading:
# https://github.com/ourresearch/openalex-institution-parsing/tree/main/V1
PREDICTOR = Predictor('')
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        _ = PREDICTOR.basic_model.get_layer('cls')
        status = 200
    except:
        status = 400
    return flask.Response(response=json.dumps(' '), status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json)
    input_df = pd.read_json(input_json, orient='records').reset_index()

    # Tokenize data
    final_df = PREDICTOR.raw_data_to_predictions(
        input_df, lang_thresh=0.99, basic_thresh=0.75)

    # Transform predicted labels into a list of dictionaries
    all_tags = []
    affiliation_ids, ror_ids = list(final_df['affiliation_id']), list(final_df['ror_id'])
    for (affiliation_id, ror_id) in zip(affiliation_ids, ror_ids):
        if affiliation_id != -1:
            all_tags.append({'affiliation_id': affiliation_id, 'ror_id': ror_id})
        else:
            all_tags.append({'affiliation_id': None, 'ror_id': None})
    # Transform predictions to JSON
    result = json.dumps(all_tags)
    return flask.Response(response=result, status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run()
