import sys
sys.path.append("../movielingo/")

import numpy as np
import nltk
from nltk import bigrams
from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktLanguageVars

from movielingo.text_utils import get_corrected_word
from movielingo.text_utils import get_pos_tags
from movielingo.text_utils import tokenize_words


class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '•', '\n')

BROWN_BIGRAMS = FreqDist(bigrams(brown.words(categories = ['reviews'])))
TOKENIZER = RegexpTokenizer(r'\w+')
SENT_TOKENIZER = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())




class SingleTextProcessor(object):

    """Class that stores and processes a single text"""

    def __init__(self, text_string, toeic_score, text_id):

        self.raw_text = text_string
        self.toeic_score = toeic_score
        self.text_id = text_id


    def find_mean_bigram_freq(self):

        """Actually returns the mean of the log of the bigram frequencies

        This method sets the mean_bigram_freq attribute.
        """

        bigrams = nltk.bigrams(self.words)
    
        accumulator = 0
        counter = 0
        for bigram in bigrams:
            counter += 1
            freq_bigram = BROWN_BIGRAMS.freq(bigram)
            if freq_bigram > 0:
               accumulator += np.log(freq_bigram) 
        
        if counter != 0:
            self.mean_bigram_freq = accumulator/counter
        else:
            self.mean_bigram_freq = np.nan


    def get_sentences_info(self):
        sent_from_learner = sent_tokenize(self.raw_text)
        sent_lengths = []
        sent_ids = []
        n_finite_verbs = []
        for i, sent in enumerate(sent_from_learner):
            words_in_sent = TOKENIZER.tokenize(sent)
            n_wrds_per_sent = len(words_in_sent)
            pos_tags = get_pos_tags(words_in_sent)
            finite_verbs = [pos_tag for pos_tag in pos_tags if pos_tag in ['VBD', 'VBP', 'VBZ']]
            n_finite_verbs.append(len(finite_verbs))
            sent_lengths.append(n_wrds_per_sent)

        
        self.n_finite_verbs = np.mean(n_finite_verbs)
        self.num_sents = len(sent_from_learner)
        self.mean_sent_len = np.mean(sent_lengths)
        self.median_sent_len = np.median(sent_lengths)
        self.sd_sent_len = np.std(sent_lengths)


    def process_self(self):
        words_uncorrected = tokenize_words(self.raw_text)
        self.words = [get_corrected_word(word) for word in words_uncorrected]
        self.text_len = len(self.words)
        self.pos_tags = get_pos_tags(self.words)

        self.find_mean_bigram_freq()
        self.get_sentences_info()


    def to_dict(self):
        local_dict = {}
        local_dict['text_id'] = self.text_id
        local_dict['token'] =  self.words
        local_dict['pos'] =  self.pos_tags
        local_dict['bigram_freq'] =  self.mean_bigram_freq
        local_dict['mean_sent_len'] = self.mean_sent_len
        local_dict['median_sent_len'] =  self.median_sent_len
        local_dict['sd_sent_len'] =   self.sd_sent_len
        local_dict['num_sent'] =   self.num_sents # number of sents per text
        local_dict['n_finite_verbs'] =   self.n_finite_verbs
        local_dict['L2_proficiency'] = self.toeic_score
        local_dict['text_id'] =  self.text_id
        return(local_dict)

if __name__ == "__main__":

    stp = SingleTextProcessor(text_string="This is a text string.",
        toeic_score=990, text_id=2)
    stp.process_self()
    test_dict = stp.to_dict()
    print(test_dict)