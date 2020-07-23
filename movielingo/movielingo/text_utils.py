
from collections import Counter

import numpy as np

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

#from spellchecker import SpellChecker
from spellchecker import SpellChecker

def division(x,y):
    if x == np.nan or y == np.nan:
        return np.nan
    elif float(x)==0 or float(y)==0:
        return 0
    else:
        return float(x)/float(y)

LEM = WordNetLemmatizer()
FREQS_LEMMAS = FreqDist([LEM.lemmatize(w.lower()) 
                         for w in brown.words()]) # can have errors in lemmas <= wrong postag
FREQS_TOKENS = FreqDist([w.lower() 
                         for w in brown.words()]) 
TOKENIZER = RegexpTokenizer(r'\w+')
TAG_DICT = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}
SPELL = SpellChecker()
STOP_WORDS=set(stopwords.words("english"))
          
def get_corrected_word(word):
    unknown = list(SPELL.unknown([word]))
    if len(unknown) > 0:
        return SPELL.correction(unknown[0])
    else:
        return word


def get_lemma(text_dict):
    text_dict["token"] = [str(x) for x in text_dict["token"]]
    lemma_column = get_lemma_word(text_dict)
    return lemma_column

def get_lemma_word(text_dict):
    lemma_word = [0 for _ in text_dict["token"]]
    for idx, token in enumerate(text_dict["token"]):
        loc_pos = text_dict["pos"][idx][0].upper()
        wordnet_pos = get_wordnet_pos(token, loc_pos)
        lemma_word[idx] = LEM.lemmatize(token, wordnet_pos)
    return lemma_word

def get_wordnet_pos(word, tag):
    """Map POS tag to first character lemmatize() accepts"""
    return TAG_DICT.get(tag, wordnet.NOUN)

def get_token_frequency(text_dict):
    wf_token = [0 for _ in text_dict["token"]]
    for idx, token in enumerate(text_dict["token"]):
        ft = FREQS_TOKENS.freq(token) 
        if ft > 0 :
            wf_token[idx] = np.log(ft)
        else:
            wf_token[idx] = np.nan

    return wf_token

def get_lemma_frequency(text_dict):
    wf_lemma = [0 for _ in text_dict["lemma"]]
    for idx, lemma in enumerate(text_dict["lemma"]):
        ft = FREQS_LEMMAS.freq(lemma) 
        if ft > 0 :
            wf_lemma[idx] = np.log(ft)
        else:
            wf_lemma[idx] = np.nan

    return wf_lemma

def get_is_stopword(text_dict):
    is_stopword_column = [i in STOP_WORDS for i in text_dict["token"]]
    return is_stopword_column

def get_pos_tags(words):
    pos_tags = [word_and_pos[1] for word_and_pos in pos_tag(words)]
    return pos_tags

def tokenize_words(text_string):
    words = [w.lower() for w in TOKENIZER.tokenize(text_string)]
    return words

def get_rare_lemma_frequency(text_dict):
    n_unique_lemmas_abs = len(Counter(text_dict["lemma"]).keys())
    n_unique_rare_lemmas = len(Counter([i == 0 for i in text_dict["lemma"]]).keys())
    return division(n_unique_rare_lemmas, n_unique_lemmas_abs)

def get_n_unique_lemmas(text_dict):
    text_length = len(text_dict)
    n_unique_lemmas_abs = len(Counter(text_dict["lemma"]).keys())
    n_unique_lemmas = division(n_unique_lemmas_abs, text_length)
    return n_unique_lemmas
    
def get_n_unique_verb_forms(text_dict):
    '''Get the number of unique verb-based part of speech tags'''
    text_length = len(text_dict["token"])
    n_uniq_verb_forms_abs = 0

    unique_pos = Counter(text_dict["pos"]).keys()
    for unipos in unique_pos:
        if unipos.startswith('V'):
            n_uniq_verb_forms_abs += 1

    n_uniq_verb_forms = division(n_uniq_verb_forms_abs, text_length)
    return n_uniq_verb_forms

def get_n_unique_past_verbs(text_dict):
    '''Get the number of unique tokens among past tense verbs'''
    text_length = len(text_dict["token"])
    past_verbs_abs = []

    for idx, tok in enumerate(text_dict["token"]):
        if text_dict["pos"][idx] in ['VBD', 'VBN']:
            past_verbs_abs.append(tok)

    n_unique_past_verbs_abs = len(Counter(past_verbs_abs).keys())
    n_unique_past_verbs = division(n_unique_past_verbs_abs, text_length)
    return n_unique_past_verbs

def get_n_unique_adjectives(text_dict):
    text_length = len(text_dict["token"])
    adj_abs = []

    for idx, tok in enumerate(text_dict["token"]):
        if text_dict["pos"][idx].startswith('J'):
            adj_abs.append(tok)

    n_unique_adj_abs = len(Counter(adj_abs).keys())
    n_unique_adj = division(n_unique_adj_abs, text_length)
    return n_unique_adj

def get_n_unique_adverbs(text_dict):
    text_length = len(text_dict["token"])
    adv_abs = []

    for idx, tok in enumerate(text_dict["token"]):
        if text_dict["pos"][idx] in ['RB']:
            adv_abs.append(tok)

    n_unique_adv_abs = len(Counter(adv_abs).keys())
    n_unique_adv = division(n_unique_adv_abs, text_length)

    return n_unique_adv

def get_n_unique_prepos(text_dict):
    
    prepos = []
    for idx, tok in enumerate(text_dict["token"]):
        if text_dict["pos"][idx] == 'IN':
            prepos.append(tok)

    n_unique_prepos = len(Counter(prepos).keys())
    return n_unique_prepos
    
def get_n_unique_modals(text_dict):
    modals = []
    for idx, tok in enumerate(text_dict["token"]):
        if text_dict["pos"][idx] == 'MD':
            modals.append(tok)

    n_unique_modals = len(Counter(modals).keys())
    return n_unique_modals
    
def get_n_wh(text_dict):
    '''Get the number of wh- words (whom/which/why etc)'''
    whs = 0
    for idx, tok in enumerate(text_dict["token"]):
        if text_dict["pos"][idx].startswith('W'):
            whs += 1 
    return whs