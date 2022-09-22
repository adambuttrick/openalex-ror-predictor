import os
import re
import json
import flask
import pickle
import time
import unidecode
from utils import load_pkl, invert_dict
from langdetect import detect
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

class Predictor:
    def __init__(self, path_to_model):
        self.model_path = path_to_model
        self.departments_list = load_pkl(self.model_path, 'departments_list.pkl')
        self.full_affiliation_dict = load_pkl(
            self.model_path, 'full_affiliation_dict.pkl')
        self.countries_list_flat = load_pkl(
            self.model_path, 'countries_list_flat.pkl')
        self.city_country_list = load_pkl(self.model_path, 'city_country_list.pkl')
        self.affiliation_vocab_basic = load_pkl(
            self.model_path, 'affiliation_vocab_basic.pkl')
        self.affiliation_vocab_language = load_pkl(
            self.model_path, 'language_model/vocab.pkl')
        self.institution_ror_id_mapping = load_pkl(self.model_path,
                                                   'institution_ror_id_mapping.pkl')
        self.countries_dict = json.load(
            open(os.path.join(self.model_path, "countries.json"), "r"))
        self.inverse_affiliation_vocab_basic = invert_dict(
            self.affiliation_vocab_basic)
        self.inverse_affiliation_vocab_language = invert_dict(
            self.affiliation_vocab_language)

    # Load the tokenizers
        self.language_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", return_tensors='tf')
        self.data_collator = DataCollatorWithPadding(tokenizer=self.language_tokenizer,
                                                     return_tensors='tf')
        self.basic_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(self.model_path, "basic_model_tokenizer"))

        # Load the models
        self.language_model = TFAutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.model_path, "language_model"))
        self.language_model.trainable = False
        self.basic_model = tf.keras.models.load_model(
            os.path.join(self.model_path, "basic_model"), compile=False)
        self.basic_model.trainable = False

    def institution_id_to_ror_id(self, institution_id):
        try:
            return self.institution_ror_id_mapping[str(institution_id)]
        except KeyError:
            return None

    def get_country_in_string(self, text):
        """
        Looks for countries in the affiliation string to be used in filtering later on.
        """
        countries_in_string = []
        for country_name, aliases in self.countries_dict.items():
            country_in_text = False
            for alias in aliases:
                if re.search(fr"\b{alias}\b", text):
                    country_in_text = True
                elif re.search(fr"\b{alias}\b", text.replace(".", "")):
                    country_in_text = True
            if country_in_text == True:
                countries_in_string.append(country_name)
        return list(set(countries_in_string))

    def max_len_and_pad(self, tok_sent):
        """
        Processes the basic model data to the correct input length.
        """
        max_len = 128
        tok_sent = tok_sent[:max_len]
        tok_sent = tok_sent + [0]*(max_len - len(tok_sent))
        return tok_sent

    def get_language(self, orig_aff_string):
        """
        Guesses the language of the affiliation string to be used for filtering later.
        """
        try:
            string_lang = detect(orig_aff_string)
        except:
            string_lang = 'en'
        return string_lang

    def get_initial_pred(self, orig_aff_string, string_lang, countries_in_string, comma_split_len):
        """
        Initial hard-coded filtering of the affiliation text to ensure that meaningless strings
        and strings in other languages are not given an institution.
        """
        orig_aff_string = str(orig_aff_string)
        if string_lang in ['fa', 'ko', 'zh-cn', 'zh-tw', 'ja', 'uk', 'ru', 'vi']:
            init_pred = None
        elif not str(orig_aff_string).strip():
            init_pred = None
        elif ((orig_aff_string.startswith("Dep") |
               orig_aff_string.startswith("School") |
               orig_aff_string.startswith("Ministry")) &
              (comma_split_len < 2) &
              (not countries_in_string)):
            init_pred = None
        elif orig_aff_string in self.departments_list:
            init_pred = None
        elif orig_aff_string in self.city_country_list:
            init_pred = None
        elif re.search(r"\b(LIANG|YANG|LIU|et al|XIE|JIA|ZHANG|QU)\b",
                       orig_aff_string):
            init_pred = None
        else:
            init_pred = 0
        return init_pred

    def get_final_basic_or_language_model_pred(self, scores, labels, countries, vocab, inv_vocab):
        """
        Takes the scores and labels from either model and performs a quick country matching
        to see if the country found in the string can be matched to the country of the
        predicted institution.
        """
        mapped_labels = [inv_vocab[i]
                         for i, j in zip(labels, scores) if i != vocab[-1]]
        scores = [j for i, j in zip(labels, scores) if i != vocab[-1]]
        final_pred = mapped_labels[0]
        final_score = scores[0]
        if not self.full_affiliation_dict[mapped_labels[0]]['country']:
            pass
        else:
            if not countries:
                pass
            else:
                for pred, score in zip(mapped_labels, scores):
                    if not self.full_affiliation_dict[pred]['country']:
                        # trying pass instead of break to give time to find the correct country
                        pass
                    elif self.full_affiliation_dict[pred]['country'] in countries:
                        final_pred = pred
                        final_score = score
                        break
                    else:
                        pass
        return final_pred, final_score

    def get_language_model_prediction(self, decoded_text, all_countries):
        """
        Preprocesses the decoded text and gets the output labels and scores for the language model.
        """
        lang_tok_data = self.language_tokenizer(
            decoded_text, truncation=True, padding=True, max_length=512)

        data = self.data_collator(lang_tok_data)
        all_scores, all_labels = tf.math.top_k(tf.nn.softmax(
            self.language_model.predict([data['input_ids'],
                                         data['attention_mask']]).logits).numpy(), 5)

        all_scores = all_scores.numpy().tolist()
        all_labels = all_labels.numpy().tolist()

        final_preds_scores = []
        for scores, labels, countries in zip(all_scores, all_labels, all_countries):
            final_pred, final_score = self.get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                                  self.affiliation_vocab_language,
                                                                                  self.inverse_affiliation_vocab_language)
            final_preds_scores.append([final_pred, final_score])
        return final_preds_scores

    def get_basic_model_prediction(self, decoded_text, all_countries):
        """
        Preprocesses the decoded text and gets the output labels and scores for the basic model.
        """
        basic_tok_data = self.basic_tokenizer(decoded_text)['input_ids']
        basic_tok_data = [self.max_len_and_pad(x) for x in basic_tok_data]
        basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)
        all_scores, all_labels = tf.math.top_k(
            self.basic_model.predict(basic_tok_data), 5)

        all_scores = all_scores.numpy().tolist()
        all_labels = all_labels.numpy().tolist()

        final_preds_scores = []
        for scores, labels, countries in zip(all_scores, all_labels, all_countries):
            final_pred, final_score = self.get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                                  self.affiliation_vocab_basic,
                                                                                  self.inverse_affiliation_vocab_basic)
            final_preds_scores.append([final_pred, final_score])
        return final_preds_scores

    def get_final_prediction(self, basic_pred_score, lang_pred_score, countries, raw_sentence, lang_thresh, basic_thresh):
        """
        Performs the model comparison and filtering to get the final prediction.
        """

        # Getting the individual preds and scores for both models
        pred_lang, score_lang = lang_pred_score
        pred_basic, score_basic = basic_pred_score

        # Logic for combining the two models
        if pred_lang == pred_basic:
            final_pred = pred_lang
            final_score = score_lang
            final_cat = 'match'
        elif score_basic > basic_thresh:
            final_pred = pred_basic
            final_score = score_basic
            final_cat = 'basic_thresh'
        elif score_lang > lang_thresh:
            final_pred = pred_lang
            final_score = score_lang
            final_cat = 'lang_thresh'
        # Inherited from OpenAlex. No idea what the edge case is here.
        elif (score_basic > 0.01) & ('China' in countries) & ('Natural Resource' in raw_sentence):
            final_pred = pred_basic
            final_score = score_basic
            final_cat = 'basic_thresh_second'
        else:
            final_pred = None
            final_score = 0.0
            final_cat = 'nothing'
        return [final_pred, final_score, final_cat]

    def raw_data_to_predictions(self, df, lang_thresh, basic_thresh):
        """
        High level function to go from a raw input dataframe to the final dataframe with affiliation
        ID prediction.
        """
        # Implementing the functions above
        df['affiliation_string'] = df['affiliation_string'].fillna(
            ' ').astype('str')
        df['lang'] = df['affiliation_string'].apply(self.get_language)
        df['country_in_string'] = df['affiliation_string'].apply(
            self.get_country_in_string)
        df['comma_split_len'] = df['affiliation_string'].apply(lambda x: len([i if i else "" for i in
                                                                              x.split(",")]))

        # Gets initial indicator of whether or not the string should go through the models
        df['affiliation_id'] = df.apply(lambda x: self.get_initial_pred(x.affiliation_string, x.lang,
                                                                        x.country_in_string, x.comma_split_len), axis=1)

        # Filter out strings that won't go through the models
        to_predict = df[df['affiliation_id'] == 0.0].drop_duplicates(
            subset=['affiliation_string']).copy()
        if to_predict.shape[0] > 0:
            to_predict['affiliation_id'] = to_predict['affiliation_id'].astype(
                'int')

            # Decode text so only ASCII characters are used
            to_predict['decoded_text'] = to_predict['affiliation_string'].apply(
                unidecode.unidecode)

            # Get predictions and scores for each model
            to_predict['lang_pred_score'] = self.get_language_model_prediction(to_predict['decoded_text'].to_list(),
                                                                               to_predict['country_in_string'].to_list())
            to_predict['basic_pred_score'] = self.get_basic_model_prediction(to_predict['decoded_text'].to_list(),
                                                                             to_predict['country_in_string'].to_list())

            # Get the final prediction for each affiliation string
            to_predict['affiliation_id'] = to_predict.apply(lambda x:
                                                            self.get_final_prediction(x.basic_pred_score,
                                                                                      x.lang_pred_score,
                                                                                      x.country_in_string,
                                                                                      x.affiliation_string,
                                                                                      lang_thresh, basic_thresh)[0], axis=1)

            # Merge predictions to original dataframe to get the same order as the data that was requested
            final_df = df[['affiliation_string']].merge(to_predict[['affiliation_string', 'affiliation_id']],
                                                        how='left', on='affiliation_string')

            final_df['affiliation_id'] = final_df['affiliation_id'].fillna(
                -1).astype('int')
            final_df['ror_id'] = [self.institution_id_to_ror_id(affiliation_id) for affiliation_id in
                final_df['affiliation_id']]
        else:
            final_df = df[['affiliation_string', 'affiliation_id']].copy()
            final_df['affiliation_id'] = final_df['affiliation_id'].fillna(
                -1).astype('int')
            final_df['ror_id'] = [self.institution_id_to_ror_id(affiliation_id) for affiliation_id in
                final_df['affiliation_id']]
        return final_df
