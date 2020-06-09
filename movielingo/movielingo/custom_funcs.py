from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist, ProbDistI
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import nps_chat
import nltk
import os, re
import numpy as np
from requests import get
from bs4 import BeautifulSoup
from math import floor
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars
from spellchecker import SpellChecker
from tqdm import tqdm

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '•', '\n')

SENT_TOKENIZER = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
STOP_WORDS=set(stopwords.words("english"))
LEM = WordNetLemmatizer()
TOKENIZER = RegexpTokenizer(r'\w+')
FREQS_LEMMAS = FreqDist([LEM.lemmatize(w.lower()) 
                         for w in nps_chat.words()]) # can have errors in lemmas <= wrong postag
FREQS_TOKENS = FreqDist([w.lower() 
                         for w in nps_chat.words()]) # can have errors in lemmas <= wrong postag
                

def create_df_with_learners_texts(list_of_text_files, directory):
    """
    Process text files, compile a dataframe 
    Features: text_id, L2_proficiency, topic, token, sent_length, pos

    :param: list of text files with L2 learners' texts separated by \n
    :returns: Pandas dataframe with shape (n_words, n_features)
    """
    l2_dict = {'text_id': [], 'L2_proficiency': [], 'topic': [], 'token': [], 'mean_sent_len': [],
    'median_sent_len': [], 'sd_sent_len': [],
    'sent_id': [], 'num_sent': [], 'pos': []}
    text_id = 1
    for file in list_of_text_files:
        filename = os.path.join(directory+file)
        #print(filename)
        lang_level = re.search("[A-C][1-2]_[0-3]", file).group()
        topic = re.search("(SMK|PTJ)", file).group()
        texts = np.genfromtxt(filename,delimiter='\n',dtype='str')
        for itext in texts:
            words, pos_tags, mean_sent_len, median_sent_len, sd_sent_len, sent_ids, num_sents, text_len = process_raw_text(itext)
            l2_dict['token'] =  l2_dict['token'] + words
            l2_dict['pos'] =  l2_dict['pos'] + pos_tags
            l2_dict['mean_sent_len'] =  l2_dict['mean_sent_len'] + fill_feature(mean_sent_len, text_len)
            l2_dict['median_sent_len'] =  l2_dict['median_sent_len'] + fill_feature(median_sent_len, text_len)
            l2_dict['sd_sent_len'] =  l2_dict['sd_sent_len'] + fill_feature(sd_sent_len, text_len)
            l2_dict['sent_id'] =  l2_dict['sent_id'] + sent_ids # sentence order number within text
            l2_dict['num_sent'] =  l2_dict['num_sent'] + fill_feature(num_sents, text_len) # number of sents per text
            l2_dict['topic'] =  l2_dict['topic'] + fill_feature(topic, text_len)
            l2_dict['L2_proficiency'] = l2_dict['L2_proficiency'] + fill_feature(lang_level, text_len)
            l2_dict['text_id'] =  l2_dict['text_id'] + fill_feature(text_id, text_len)
            text_id += 1
    df = pd.DataFrame.from_dict(l2_dict)
    df['pos'] = df.apply(lambda row: 'UH' if row.pos == '$' else row.pos, axis =1)
    df['pos'] = df.apply(lambda row: 'NN' if row.pos == "''" else row.pos, axis =1)
    return df

def create_one_movie_df(imdb_id, directory):
    """
    Process text files, compile a dataframe 
    Features: text_id, token, sent_length, pos

    :param: list of text files with L2 learners' texts separated by \n
    :returns: Pandas dataframe with shape (n_words, n_features)
    """
    movie_dict = {'text_id': [], 'token': [], 'mean_sent_len': [],
    'median_sent_len': [], 'sd_sent_len': [],
    'sent_id': [], 'num_sent': [], 'pos': []}
    text_id = 1
    files = os.listdir(directory)
    list_of_text_files = []
    for file in files:
        if re.search(imdb_id, file):
            list_of_text_files.append(file)
    for file in list_of_text_files:
        filename = os.path.join(directory+file)
        with open(filename, 'r') as subtitles:
            texts = subtitles.read()
        sents = SENT_TOKENIZER.tokenize(texts)
        for itext in range(floor(len(sents)/5)):
            text_window = sents[itext*5:(itext+1)*5]
            text_window_raw = TreebankWordDetokenizer().detokenize(text_window)
            words, pos_tags, mean_sent_len, median_sent_len, sd_sent_len, sent_ids, num_sents, text_len = process_raw_text(text_window_raw)
            movie_dict['token'] =  movie_dict['token'] + words
            movie_dict['pos'] =  movie_dict['pos'] + pos_tags
            movie_dict['mean_sent_len'] =  movie_dict['mean_sent_len'] + fill_feature(mean_sent_len, text_len)
            movie_dict['median_sent_len'] =  movie_dict['median_sent_len'] + fill_feature(median_sent_len, text_len)
            movie_dict['sd_sent_len'] =  movie_dict['sd_sent_len'] + fill_feature(sd_sent_len, text_len)
            movie_dict['sent_id'] =  movie_dict['sent_id'] + sent_ids # sentence order number within text
            movie_dict['num_sent'] =  movie_dict['num_sent'] + fill_feature(num_sents, text_len) # number of sents per text
            movie_dict['text_id'] =  movie_dict['text_id'] + fill_feature(text_id, text_len)
            text_id += 1
    df = pd.DataFrame.from_dict(movie_dict)
    return df


    
def create_df_with_gachon_texts(file, directory): 
    """
    Process text files, compile a dataframe 
    Features: text_id, L2_proficiency, topic, token, sent_length, pos

    :param: list of text files with L2 learners' texts separated by \n
    :returns: Pandas dataframe with shape (n_words, n_features)
    """
    l2_dict = {'text_id': [], 'L2_proficiency': [], 'token': [], 'mean_sent_len': [],
    'median_sent_len': [], 'sd_sent_len': [],
    'sent_id': [], 'num_sent': [], 'pos': []}

    filename = os.path.join(directory+file)
    texts = pd.read_csv(filename)
    for counter, itext in enumerate(tqdm(texts.text)):
        words, pos_tags, mean_sent_len, median_sent_len, sd_sent_len, sent_ids, num_sents, text_len = process_raw_text(itext)
        l2_dict['token'] =  l2_dict['token'] + words
        l2_dict['pos'] =  l2_dict['pos'] + pos_tags
        l2_dict['mean_sent_len'] =  l2_dict['mean_sent_len'] + fill_feature(mean_sent_len, text_len)
        l2_dict['median_sent_len'] =  l2_dict['median_sent_len'] + fill_feature(median_sent_len, text_len)
        l2_dict['sd_sent_len'] =  l2_dict['sd_sent_len'] + fill_feature(sd_sent_len, text_len)
        l2_dict['sent_id'] =  l2_dict['sent_id'] + sent_ids # sentence order number within text
        l2_dict['num_sent'] =  l2_dict['num_sent'] + fill_feature(num_sents, text_len) # number of sents per text
        l2_dict['L2_proficiency'] = l2_dict['L2_proficiency'] + fill_feature(str(texts.toeic[counter]), text_len)
        l2_dict['text_id'] =  l2_dict['text_id'] + fill_feature(str(texts.student[counter]), text_len)


    df = pd.DataFrame.from_dict(l2_dict)
    #df['pos'] = df.apply(lambda row: 'UH' if row.pos == '$' else row.pos, axis =1)
    #df['pos'] = df.apply(lambda row: 'NN' if row.pos == "''" else row.pos, axis =1)
    return df



# def get_imdb_id(movie_title):
#     text_id = get_movie_title(imdb_id)
#         #time.sleep(1)

def fill_feature(text_feature, text_length):
    """
    Repeat text feature as many times as there are words (to fill dataframe)
    ex: input: text_id of 3-word long text, output: [text_id, text_id, text_id]
    """
    return np.repeat(text_feature, text_length).tolist()
    
def process_raw_text(text_string):
    """
    Processes a single chunk of raw text:
    :param: text string
    :returns: words, POS tags, sentence_length, text_length per word as lists
    """
    words = tokenize_words(text_string)
    text_len = len(words)
    pos_tags = get_pos_tags(words)
    mean_sent_len, median_sent_len, sd_sent_len, sent_ids, num_sents = get_sentence_info(text_string)
    return words, pos_tags, mean_sent_len, median_sent_len, sd_sent_len, sent_ids, num_sents, text_len
    
def tokenize_words(text_string):
    words = [w.lower() for w in TOKENIZER.tokenize(text_string)]
    return words

def get_pos_tags(words):
    pos_tags = [word_and_pos[1] for word_and_pos in nltk.pos_tag(words)]
    return pos_tags
    
def get_sentence_info(text_string):
    sent_from_learner = sent_tokenize(text_string)
    sent_lengths = []
    sent_ids = []
    for sent in sent_from_learner:
        n_wrds_per_sent = len(TOKENIZER.tokenize(sent))
        current_sent_len = fill_feature(n_wrds_per_sent, n_wrds_per_sent)
        sent_lengths = sent_lengths + [n_wrds_per_sent]
        sent_ids = sent_ids + fill_feature(sent, n_wrds_per_sent)
    num_sents = len(sent_from_learner)
    mean_sent_len = np.mean(sent_lengths)
    median_sent_len = np.median(sent_lengths)
    sd_sent_len = np.std(sent_lengths)
    return mean_sent_len, median_sent_len, sd_sent_len, sent_ids, num_sents

def engineer_features(df, data_type):
    """
    Add linguistic features to the dataframe with single words/sentences
    :param: df with columns 'text_id', 'token', 
    :returns: features_df
    """
    df['lemma'] = get_lemma(df, 'token', 'pos') # mean by text
    df['wf_lemma'] = get_lemma_frequency(df, 'lemma') # mean by text
    df['wf_token'] = get_token_frequency(df, 'token') # mean by text
    df['is_stopword'] = get_is_stopword(df, 'token') # to exclude stopwords
    df = df[df['is_stopword'] == False].copy().reset_index(drop=True)
    if data_type == 'movie':
        features = df.groupby(['text_id']).count().reset_index()[['text_id']]
    else:
        features = df.groupby(['text_id', 'L2_proficiency']).count().reset_index()[['text_id', 'L2_proficiency']]
    features['n_uniq_lemmas'] = get_n_unique_lemmas(df, 'text_id', 'lemma') # in text
    features['n_interjections'] = get_n_interjections(df, 'text_id', 'pos')
    features['mean_sent_len'], features['median_sent_len'], features['sd_sent_len'] = aggr_sent_features(df, 'text_id')
    features['wf_lemma'] = df.groupby('text_id').wf_lemma.mean().reset_index().wf_lemma
    features['wf_token'] = df.groupby('text_id').wf_token.mean().reset_index().wf_token
    features['n_unique_verb_forms'] = get_n_unique_verb_forms(df, 'text_id', 'pos')
    return df, features
    
def get_wordnet_pos(word, tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def get_lemma(df, token_column, pos_column):
    lemma_column = df.apply(lambda row:
    LEM.lemmatize(row[token_column], get_wordnet_pos(row[token_column], row[pos_column][0].upper())),
                    axis = 1)
    return lemma_column
    
def get_lemma_frequency(df, lemma_column):
    wf_lemma_column = df.apply(lambda row: FREQS_LEMMAS.freq(row[lemma_column])
                          if FREQS_LEMMAS.freq(row[lemma_column]) > 0 else 0, axis = 1)
    return wf_lemma_column
    
def get_token_frequency(df, token_column):
    wf_token_column = df.apply(lambda row: FREQS_TOKENS.freq(row[token_column])
                          if FREQS_TOKENS.freq(row[token_column]) > 0 else 0, axis = 1)
    return wf_token_column
    
def get_is_stopword(df, token_column):
    is_stopword_column = df.apply(lambda row: row[token_column] in STOP_WORDS, axis = 1)
    return is_stopword_column
    
def get_n_unique_lemmas(df, text_id, lemma_column):
    n_unique_lemmas = df.groupby(text_id)[lemma_column].nunique().reset_index().lemma
    return n_unique_lemmas
    
def get_n_interjections(df, text_id, pos_column):
    n_interjections = df.loc[df[pos_column] == 'UH'].groupby(text_id).token.count().reset_index().token
    return n_interjections

def aggr_sent_features(df, text_id):
    mean_sent_len = df.groupby(text_id).mean_sent_len.mean().reset_index().mean_sent_len
    median_sent_len = df.groupby(text_id).median_sent_len.mean().reset_index().median_sent_len
    sd_sent_len = df.groupby(text_id).sd_sent_len.mean().reset_index().sd_sent_len
    return mean_sent_len, median_sent_len, sd_sent_len

def get_n_unique_verb_forms(df, text_id, pos_column):
    n_uniq_verb_forms = df.loc[df[pos_column].str.startswith('V', na = False)].groupby(text_id)[pos_column].nunique().reset_index()[pos_column]
    return n_uniq_verb_forms
#     get_n_unique_verb_forms() # in text
#     get_n_unique_adjectives() # in text (if for movie sliding window we use the mean length of these texts)
#     get_n_unique_adverbs()
#     get_n_unique_prepos()
#     get_n_unique_modals()
#    get_n_wh()

def get_movie_title(imdb_id):
    url = 'https://www.imdb.com/title/tt' + imdb_id + '/'
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    return html_soup.title.text.strip(' - IMDb')