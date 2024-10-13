import os
import re
import json
import pickle
from unidecode import unidecode
from collections import Counter
from langdetect import detect
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast


class AffiliationRecord:
    def __init__(self, affiliation_string):
        self.affiliation_string = affiliation_string
        self.lang = None
        self.country_in_string = None
        self.comma_split_len = None
        self.institution_id = None
        self.decoded_text = None
        self.lang_pred_score = None
        self.basic_pred_score = None
        self.ror_id = None
        self.score = None
        self.category = None


class InstitutionTagger:
    def __init__(self, model_path="institution_tagger_v2_artifacts"):
        self._load_data(model_path)
        self._initialize_models(model_path)

    def _load_data(self, model_path):
        with open(os.path.join(model_path, "departments_list.pkl"), "rb") as f:
            self.departments_list = pickle.load(f)

        with open(os.path.join(model_path, "full_affiliation_dict.pkl"), "rb") as f:
            self.full_affiliation_dict = pickle.load(f)

        with open(os.path.join(model_path, "multi_inst_names_ids.pkl"), "rb") as f:
            self.multi_inst_names_ids = pickle.load(f)

        with open(os.path.join(model_path, "countries_list_flat.pkl"), "rb") as f:
            self.countries_list_flat = pickle.load(f)

        with open(os.path.join(model_path, "countries.json"), "r") as f:
            self.countries_dict = json.load(f)

        with open(os.path.join(model_path, "city_country_list.pkl"), "rb") as f:
            self.city_country_list = pickle.load(f)

        with open(os.path.join(model_path, "affiliation_vocab.pkl"), "rb") as f:
            self.affiliation_vocab = pickle.load(f)

        self.inverse_affiliation_vocab = {
            i: j for j, i in self.affiliation_vocab.items()}

        with open(os.path.join(model_path, "institutions.pkl"), "rb") as f:
            self.institution_ror_mapping = pickle.load(f)

    def _initialize_models(self, model_path):
        self.language_tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", return_tensors='tf')
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.language_tokenizer, return_tensors='tf')

        self.basic_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(model_path, "basic_model_tokenizer"))

        self.language_model = TFAutoModelForSequenceClassification.from_pretrained(
            os.path.join(model_path, "language_model")
        )
        self.language_model.trainable = False

        self.basic_model = tf.keras.models.load_model(
            os.path.join(model_path, "basic_model"), compile=False)
        self.basic_model.trainable = False

    def _map_institution_to_ror(self, institution_id):
        return self.institution_ror_mapping.get(institution_id, None)

    def _string_match_clean(self, text):
        if not isinstance(text, str):
            return ""

        # Replace "&" with "and"
        if "r&d" not in text.lower():
            text = text.replace(" & ", " and ")

        # Remove country
        if text.strip().endswith(")"):
            for country in self.countries_list_flat:
                if text.strip().endswith(f"({country})"):
                    text = text.replace(f"({country})", "")

        # Use unidecode
        text = unidecode(text.strip())

        # Replace common abbreviations
        text = text.replace("Univ.", "University")
        text = text.replace("Lab.", "Laboratory")
        text = text.replace("U.S. Army", "United States Army")
        text = text.replace("U.S. Navy", "United States Navy")
        text = text.replace("U.S. Air Force", "United States Air Force")

        # Remove special characters
        text = re.sub("[^0-9a-zA-Z]", "", text)

        return text

    def _get_country_in_string(self, text):
        """
        Looks for countries in the affiliation string to be used in filtering later on.
        """
        countries_in_string = []
        if not isinstance(text, str):
            print(f"Warning: Non-string input received in _get_country_in_string: {text}")
            return countries_in_string

        try:
            for x, y in self.countries_dict.items():
                if np.max([1 if re.search(fr"\b{i}\b", text) else 0 for i in y]) > 0:
                    countries_in_string.append(x)

            for x, y in self.countries_dict.items():
                if np.max([1 if re.search(fr"\b{i}\b", text.replace(".", "")) else 0 for i in y]) > 0:
                    countries_in_string.append(x)
        except Exception as e:
            print(f"Error processing text in _get_country_in_string: {text}")
            print(f"Error details: {str(e)}")

        return list(set(countries_in_string))

    def _max_len_and_pad(self, tok_sent):
        """
        Processes the basic model data to the correct input length.
        """
        max_len = 128
        tok_sent = tok_sent[:max_len]
        tok_sent = tok_sent + [0] * (max_len - len(tok_sent))
        return tok_sent

    def _get_language(self, orig_aff_string):
        """
        Guesses the language of the affiliation string to be used for filtering later.
        """
        try:
            string_lang = detect(orig_aff_string)
        except:
            string_lang = 'en'

        return string_lang

    def _get_initial_pred(self, orig_aff_string, string_lang, countries_in_string, comma_split_len):
        """
        Initial hard-coded filtering of the affiliation text to ensure that meaningless strings
        and strings in other languages are not given an institution.
        """
        if string_lang in ['fa', 'ko', 'zh-cn', 'zh-tw', 'ja', 'uk', 'ru', 'vi', 'ar']:
            init_pred = None
        elif len(self._string_match_clean(str(orig_aff_string))) <= 2:
            init_pred = None
        elif (
            (
                orig_aff_string.startswith("Dep")
                or orig_aff_string.startswith("School")
                or orig_aff_string.startswith("Ministry")
            )
            and (comma_split_len < 2)
            and (not countries_in_string)
        ):
            init_pred = None
        elif orig_aff_string in self.departments_list:
            init_pred = None
        elif self._string_match_clean(str(orig_aff_string).strip()) in self.city_country_list:
            init_pred = None
        elif re.search(r"\b(LIANG|YANG|LIU|XIE|JIA|ZHANG)\b", orig_aff_string):
            for inst_name in [
                "Hospital",
                "University",
                "School",
                "Academy",
                "Institute",
                "Ministry",
                "Laboratory",
                "College",
            ]:
                if inst_name in str(orig_aff_string):
                    init_pred = 0
                    break
                else:
                    init_pred = None
        elif re.search(r"\b(et al)\b", orig_aff_string):
            if str(orig_aff_string).strip().endswith('et al'):
                init_pred = None
            else:
                init_pred = 0
        else:
            init_pred = 0
        return init_pred

    def _get_language_model_prediction(self, decoded_texts, all_countries):
        """
        Preprocesses the decoded text and gets the output labels and scores for the language model.
        """
        lang_tok_data = self.language_tokenizer(
            decoded_texts, truncation=True, padding=True, max_length=512)

        data = self.data_collator(lang_tok_data)
        outputs = self.language_model.predict(
            [data['input_ids'], data['attention_mask']], verbose=0)
        logits = outputs.logits
        probs = tf.nn.softmax(logits).numpy()
        all_scores, all_labels = tf.math.top_k(probs, 20)

        all_scores = all_scores.numpy().tolist()
        all_labels = all_labels.numpy().tolist()

        final_preds_scores = []
        for scores, labels, countries in zip(all_scores, all_labels, all_countries):
            final_pred, final_score, mapping = self._get_final_basic_or_language_model_pred(
                scores, labels, countries, self.affiliation_vocab, self.inverse_affiliation_vocab
            )
            final_preds_scores.append([final_pred, final_score, mapping])

        return final_preds_scores

    def _get_basic_model_prediction(self, decoded_texts, all_countries):
        """
        Preprocesses the decoded text and gets the output labels and scores for the basic model.
        """
        basic_tok_data = self.basic_tokenizer(decoded_texts)['input_ids']
        basic_tok_data = [self._max_len_and_pad(x) for x in basic_tok_data]
        basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)
        outputs = self.basic_model.predict(basic_tok_data, verbose=0)
        all_scores, all_labels = tf.math.top_k(outputs, 20)

        all_scores = all_scores.numpy().tolist()
        all_labels = all_labels.numpy().tolist()

        final_preds_scores = []
        for scores, labels, countries in zip(all_scores, all_labels, all_countries):
            final_pred, final_score, mapping = self._get_final_basic_or_language_model_pred(
                scores, labels, countries, self.affiliation_vocab, self.inverse_affiliation_vocab
            )
            final_preds_scores.append([final_pred, final_score, mapping])

        return final_preds_scores

    def _get_final_basic_or_language_model_pred(self, scores, labels, countries, vocab, inv_vocab):
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
                        pass
                    elif self.full_affiliation_dict[pred]['country'] in countries:
                        final_pred = pred
                        final_score = score
                        break
                    else:
                        pass
        return final_pred, final_score, mapped_labels

    def _get_similar_preds_to_remove(self, decoded_string, curr_preds):
        """
        Looks for organizations with similar/matching names and only predicts for one of those organizations.
        """
        preds_to_remove = []
        pred_display_names = [self.full_affiliation_dict[i]
                              ['display_name'] for i in curr_preds]
        counts_of_preds = Counter(pred_display_names)

        preds_array = np.array(curr_preds)
        preds_names_array = np.array(pred_display_names)

        for pred_name in counts_of_preds.items():
            temp_preds_to_remove = []
            to_use = []
            if pred_name[1] > 1:
                list_to_check = preds_array[preds_names_array == pred_name[0]].tolist(
                )
                for pred in list_to_check:
                    if self._string_match_clean(self.full_affiliation_dict[pred]['city']) in decoded_string:
                        to_use.append(pred)
                    else:
                        temp_preds_to_remove.append(pred)
                if not to_use:
                    to_use = temp_preds_to_remove[0]
                    preds_to_remove += temp_preds_to_remove[1:]
                else:
                    preds_to_remove += temp_preds_to_remove
            else:
                pass

        return preds_to_remove

    def _check_for_city_and_country_in_string(self, raw_sentence, countries, aff_dict_entry):
        """
        Checks for city and country and string for a common name institution.
        """
        if (aff_dict_entry['country'] in countries) and (aff_dict_entry['city'] in raw_sentence):
            return True
        else:
            return False

    def _get_final_prediction(
        self, basic_pred_score, lang_pred_score, countries, raw_sentence, lang_thresh, basic_thresh
    ):
        """
        Performs the model comparison and filtering to get the final prediction.
        """
        pred_lang, score_lang, mapped_lang = lang_pred_score
        pred_basic, score_basic, mapped_basic = basic_pred_score

        final_preds = []
        final_scores = []
        final_cats = []
        check_pred = []
        if pred_lang == pred_basic:
            final_preds.append(pred_lang)
            final_scores.append(score_lang)
            final_cats.append('model_match')
            check_pred.append(pred_lang)
        elif score_basic > basic_thresh:
            final_preds.append(pred_basic)
            final_scores.append(score_basic)
            final_cats.append('basic_thresh')
            check_pred.append(pred_basic)
        elif score_lang > lang_thresh:
            final_preds.append(pred_lang)
            final_scores.append(score_lang)
            final_cats.append('lang_thresh')
            check_pred.append(pred_lang)
        elif (score_basic > 0.01) and ('China' in countries) and ('Natural Resource' in raw_sentence):
            final_preds.append(pred_basic)
            final_scores.append(score_basic)
            final_cats.append('basic_thresh_second')
            check_pred.append(pred_basic)
        else:
            final_preds.append(-1)
            final_scores.append(0.0)
            final_cats.append('nothing')

        all_mapped = list(set(mapped_lang + mapped_basic))

        decoded_affiliation_string = self._string_match_clean(raw_sentence)
        all_mapped_strings = [self.full_affiliation_dict[i]
                              ['final_names'] for i in all_mapped]

        matched_preds = []
        matched_strings = []
        for inst_id, match_strings in zip(all_mapped, all_mapped_strings):
            if inst_id not in final_preds:
                for match_string in match_strings:
                    if match_string in decoded_affiliation_string:
                        if not self.full_affiliation_dict[inst_id]['country']:
                            matched_preds.append(inst_id)
                            matched_strings.append(match_string)
                        elif not countries:
                            if inst_id not in self.multi_inst_names_ids:
                                matched_preds.append(inst_id)
                                matched_strings.append(match_string)
                            else:
                                pass
                        elif self.full_affiliation_dict[inst_id]['country'] in countries:
                            matched_preds.append(inst_id)
                            matched_strings.append(match_string)
                        else:
                            pass
                        break
                    else:
                        pass
            else:
                pass

        # Need to check for institutions that are a subset of another institution
        skip_matching = []
        for inst_id, matched_string in zip(matched_preds, matched_strings):
            for inst_id2, matched_string2 in zip(matched_preds, matched_strings):
                if (matched_string in matched_string2) and (matched_string != matched_string2):
                    skip_matching.append(inst_id)

        if check_pred:
            for inst_id, matched_string in zip(matched_preds, matched_strings):
                for final_string in self.full_affiliation_dict[check_pred[0]]['final_names']:
                    if matched_string in final_string:
                        skip_matching.append(inst_id)

        for matched_pred in matched_preds:
            if matched_pred not in skip_matching:
                final_preds.append(matched_pred)
                final_scores.append(0.95)
                final_cats.append('string_match')

        if (final_cats[0] == 'nothing') and (len(final_preds) > 1):
            final_preds = final_preds[1:]
            final_scores = final_scores[1:]
            final_cats = final_cats[1:]

        # Check if many names belong to same organization name (different locations)
        if (final_preds[0] != -1) and (len(final_preds) > 1):
            final_display_names = [
                self.full_affiliation_dict[x]['display_name'] for x in final_preds]

            if len(final_display_names) == len(set(final_display_names)):
                pass
            else:
                final_preds_after_removal = []
                final_scores_after_removal = []
                final_cats_after_removal = []
                preds_to_remove = self._get_similar_preds_to_remove(
                    decoded_affiliation_string, final_preds)
                for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                    if temp_pred in preds_to_remove:
                        pass
                    else:
                        final_preds_after_removal.append(temp_pred)
                        final_scores_after_removal.append(temp_score)
                        final_cats_after_removal.append(temp_cat)

                final_preds = final_preds_after_removal
                final_scores = final_scores_after_removal
                final_cats = final_cats_after_removal
                
        preds_to_remove = []
        if final_preds[0] == -1:
            pass
        else:
            final_department_name_ids = [
                [x, str(self.full_affiliation_dict[x]['display_name'])]
                for x in final_preds
                if (
                    str(self.full_affiliation_dict[x]['display_name']).startswith(
                        "Department of")
                    or str(self.full_affiliation_dict[x]['display_name']).startswith("Department for")
                )
            ]
            if final_department_name_ids:
                for temp_id in final_department_name_ids:
                    if self._string_match_clean(temp_id[1]) not in self._string_match_clean(str(raw_sentence).strip()):
                        preds_to_remove.append(temp_id[0])
                    elif not self._check_for_city_and_country_in_string(
                        raw_sentence, countries, self.full_affiliation_dict[temp_id[0]]
                    ):
                        preds_to_remove.append(temp_id[0])
                    else:
                        pass

            if any(x in final_preds for x in self.multi_inst_names_ids):
                if len(final_preds) == 1:
                    pred_name = str(
                        self.full_affiliation_dict[final_preds[0]]['display_name'])
                    # Check if it is exact string match
                    if (
                        self._string_match_clean(pred_name)
                        == self._string_match_clean(str(raw_sentence).strip())
                    ):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['']
                    elif pred_name.startswith("Department of"):
                        if ("College" in raw_sentence) or ("University" in raw_sentence):
                            final_preds = [-1]
                            final_scores = [0.0]
                            final_cats = ['']
                        elif (
                            self._string_match_clean(
                                str(
                                    self.full_affiliation_dict[final_preds[0]]['display_name'])
                            )
                            not in self._string_match_clean(str(raw_sentence).strip())
                        ):
                            final_preds = [-1]
                            final_scores = [0.0]
                            final_cats = ['']

                else:
                    non_multi_inst_name_preds = [
                        x for x in final_preds if x not in self.multi_inst_names_ids
                    ]
                    if len(non_multi_inst_name_preds) > 0:
                        for temp_pred, temp_score, temp_cat in zip(
                            final_preds, final_scores, final_cats
                        ):
                            if temp_pred not in non_multi_inst_name_preds:
                                aff_dict_temp = self.full_affiliation_dict[temp_pred]
                                if aff_dict_temp['display_name'].startswith("Department of"):
                                    if ("College" in raw_sentence) or ("University" in raw_sentence):
                                        preds_to_remove.append(temp_pred)
                                    elif (
                                        self._string_match_clean(
                                            str(
                                                self.full_affiliation_dict[temp_pred]['display_name'])
                                        )
                                        not in self._string_match_clean(str(raw_sentence).strip())
                                    ):
                                        preds_to_remove.append(temp_pred)
                                    else:
                                        if self._check_for_city_and_country_in_string(
                                            raw_sentence, countries, aff_dict_temp
                                        ):
                                            pass
                                        else:
                                            preds_to_remove.append(temp_pred)
                                elif aff_dict_temp['country'] in countries:
                                    pass
                                else:
                                    preds_to_remove.append(temp_pred)
                    else:
                        pass
            else:
                pass

        true_final_preds = [
            x for x, y, z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove
        ]
        true_final_scores = [
            y for x, y, z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove
        ]
        true_final_cats = [
            z for x, y, z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove
        ]

        if not true_final_preds:
            true_final_preds = [-1]
            true_final_scores = [0.0]
            true_final_cats = ['']
        return [true_final_preds, true_final_scores, true_final_cats]

    def _prepare_results(self):
        results = []
        for record in self.affiliations:
            # Filter out null predictions (-1 and resulting None values) returned from the model
            def filter_valid(lst):
                return [item for item in lst if item not in (-1, None)]
            valid_institution_ids = filter_valid(record.institution_id) if isinstance(
                record.institution_id, list) else []
            valid_ror_ids = filter_valid(record.ror_id) if isinstance(
                record.ror_id, list) else []
            valid_scores = record.score[:len(valid_institution_ids)] if isinstance(
                record.score, list) else []
            valid_categories = record.category[:len(valid_institution_ids)] if isinstance(
                record.category, list) else []
            result = {
                'affiliation_string': record.affiliation_string,
                'institution_id': valid_institution_ids,
                'score': valid_scores,
                'category': valid_categories,
                'ror_id': valid_ror_ids
            }
            results.append(result)
        return results

    def predict(self, affiliation_strings, lang_thresh=0.9, basic_thresh=0.9):
        self.affiliations = [AffiliationRecord(
            aff_string) for aff_string in affiliation_strings]

        for record in self.affiliations:
            record.lang = self._get_language(record.affiliation_string)
            record.country_in_string = self._get_country_in_string(
                record.affiliation_string)
            record.comma_split_len = len(
                [i for i in record.affiliation_string.split(",") if i])
            record.institution_id = self._get_initial_pred(
                record.affiliation_string, record.lang, record.country_in_string, record.comma_split_len
            )

        to_predict = [
            record for record in self.affiliations if record.institution_id == 0]

        for record in to_predict:
            record.decoded_text = unidecode(record.affiliation_string)
            record.lang_pred_score = self._get_language_model_prediction(
                [record.decoded_text], [record.country_in_string]
            )[0]
            record.basic_pred_score = self._get_basic_model_prediction(
                [record.decoded_text], [record.country_in_string]
            )[0]

            prediction = self._get_final_prediction(
                record.basic_pred_score,
                record.lang_pred_score,
                record.country_in_string,
                record.affiliation_string,
                lang_thresh,
                basic_thresh,
            )

            record.institution_id, record.score, record.category = prediction
            record.ror_id = [self._map_institution_to_ror(
                pred) for pred in record.institution_id]

        return self._prepare_results()
