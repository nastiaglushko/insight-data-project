
import pandas as pd
import numpy as np
import os
import re
from math import floor
from pathlib import Path

import sys
sys.path.append("../../movielingo/")

from movielingo.config import subtitle_dir, processed_data_dir
from movielingo.single_text_processor import SingleTextProcessor

from movielingo.text_utils import get_n_wh
from movielingo.text_utils import get_n_unique_prepos
from movielingo.text_utils import get_n_unique_modals
from movielingo.text_utils import get_n_unique_adverbs
from movielingo.text_utils import get_n_unique_adjectives
from movielingo.text_utils import get_n_unique_past_verbs
from movielingo.text_utils import get_n_unique_verb_forms
from movielingo.text_utils import get_is_stopword
from movielingo.text_utils import get_lemma
from movielingo.text_utils import get_lemma_frequency
from movielingo.text_utils import get_token_frequency
from movielingo.text_utils import get_n_unique_lemmas
from movielingo.text_utils import get_rare_lemma_frequency

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.tokenize.treebank import TreebankWordDetokenizer

import tqdm

from multiprocessing import Pool

KEYS = ['mls', 'mlt', 'mlc', 'c_s', 'vp_t', 'c_t', 'cp_t', 'cp_c', 
        'cn_t', 'cn_c', 'ct_t', 't_s', 'dc_t', 'dc_c']

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '•', '...')

_TEXTS_TO_EXCLUDE = [3517] # text only has one letter in it
SENT_TOKENIZER = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())

def process_one_text(arguments):
    raw_text, toeic_score_str, student_id_str, mode = arguments
    rt = SingleTextProcessor(raw_text, toeic_score_str , student_id_str, mode)
    rt.process_self()
    l2_dict = rt.to_dict()
    features = engineer_features(l2_dict)
    return(features)

def create_df_from_texts(filename): 
    """
    Process multiple text files from L2 learners,
    extract features (see engineer_features()),
    return dataframe.

    :param: csv file with each text stored on a separate row of the 'text' column,
    proficiency stored in 'toeic', student_id in 'student'
    :returns: Pandas dataframe with shape (n_texts, n_features)
    """

    features_df = pd.DataFrame()
    texts = pd.read_csv(filename)

    #pool = Pool(processes=1) # parallelizing only works if dependency parsing results are not written in a file
    zip_loc = zip(texts.text, texts.toeic, texts.student)
    failures = []

    features_list = []
    for counter, (raw_text, toeic_score, text_id) in enumerate(tqdm.tqdm(zip_loc)):
        if counter%100 == 0:
            print(counter)
        if counter not in _TEXTS_TO_EXCLUDE:
            try:
                arguments = raw_text, str(toeic_score), str(text_id), 'learners'
                rt = SingleTextProcessor(*arguments)
                if len(rt.sentences) > 2:
                    rt.process_self()
                    l2_dict = rt.to_dict()
                    features = engineer_features(l2_dict)
                    features_df = features_df.append(features, ignore_index=True)
            except:
                print('Failure for counter ' + str(counter))
                failures.append(counter)
                   #features_list.append(process_one_text(arguments))
            #features_list.append(pool.apply_async(process_one_text, (arguments,)))
    
    # for features in tqdm.tqdm(features_list):
        # features_df = features_df.append(features.get(), ignore_index=True)

    return features_df, failures


def engineer_features(l2_dict):
    """
    Add linguistic features to the dictionary with single words/sentences
    :param: dictionary with keys: 'text_id','token','pos','bigram_freq','mean_sent_len',
    'median_sent_len','sd_sent_len','num_sent','n_finite_verbs','L2_proficiency'

    :returns: pandas dataframe with various linguistic features
    """
    l2_dict['lemma'] = get_lemma(l2_dict)
    l2_dict['word_len'] = len(l2_dict["token"])
    l2_dict['wf_lemma'] = get_lemma_frequency(l2_dict) # mean by text
    l2_dict['wf_token'] = get_token_frequency(l2_dict) # mean by text
    l2_dict['is_stopword'] = get_is_stopword(l2_dict) # to exclude stopwords
    l2_dict['n_uniq_rare_lemmas'] = get_rare_lemma_frequency(l2_dict)
   
    features = {"text_id": l2_dict["text_id"], "L2_proficiency": l2_dict["L2_proficiency"]}
    features['n_uniq_lemmas'] = get_n_unique_lemmas(l2_dict) # in text
    features['mean_sent_len'] = l2_dict['mean_sent_len']
    features['median_sent_len'] =  l2_dict['median_sent_len']
    features['sd_sent_len'] = l2_dict['sd_sent_len']

    features['mean_word_len'] = np.nanmean(l2_dict['word_len'])
    features['mean_wf_lemma'] = np.nanmean(l2_dict['wf_lemma'])
    features['mean_wf_token'] =  np.nanmean(l2_dict['wf_token'])
    features['mean_n_uniq_rare_lemmas'] =  np.nanmean(l2_dict['n_uniq_rare_lemmas'])

    features['n_unique_verb_forms'] = get_n_unique_verb_forms(l2_dict)
    features['n_unique_past_verbs'] = get_n_unique_past_verbs(l2_dict)
    features['n_unique_adj'] = get_n_unique_adjectives(l2_dict)
    features['n_unique_adv'] = get_n_unique_adverbs(l2_dict)
    features['n_unique_prepos'] = get_n_unique_prepos(l2_dict)
    features['n_unique_modals'] = get_n_unique_modals(l2_dict)
    features['n_wh'] = get_n_wh(l2_dict)

    for key in KEYS:
        features[key] = l2_dict[key]

    return pd.DataFrame(features, index=[0])    

if __name__ == "__main__":

    input_filename = os.path.join(processed_data_dir, '01_gachon_processed.csv')
    features_df, failures = create_df_from_texts(input_filename)
    output_filename = os.path.join(processed_data_dir, 'gachon_features.csv')
    features_df.to_csv('gachon_features.csv', index = False)