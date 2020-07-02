from bs4 import BeautifulSoup
import os
import pandas as pd
import re
import requests
import tqdm

from collections import Counter
from movielingo.config import subtitle_dir, processed_data_dir
from movielingo.batch_text_processing_multi import process_one_text
from multiprocessing import Pool
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktLanguageVars
from nltk.tokenize.treebank import TreebankWordDetokenizer

def toeic2cefr(x):
    ''' Transform TOEIC scores into CEFR language proficiency scores '''
    if x < 120/1000:
        return 'Low'
    elif x < 255/1000:
        return 'A1'
    elif x < 550/1000:
        return 'A2'
    elif x < 785/1000:
        return 'B1'
    else:
        return 'B2+'
        
class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '•', '...')

SENT_TOKENIZER = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
TEMPLATE_POSTER_URL = 'https://s.studiobinder.com/wp-content/uploads/2017/12/Movie-Poster-Template-Light-With-Image.jpg?x81279'
IMDB = pd.read_csv(processed_data_dir /'movie_details_db.csv', dtype = {'id': str})

class Movie():
    ''' A movie in the Movielingo app

    Attributes:
    - title (str)
    - IMDB ID (str)
    - link to movie poster (str)
    - subtitle features (pandas df with NLP features)
    - subtitle difficulty distribution (list)
    - IMDB page (BeautifulSoup)
    - google search results for movie (imdb href)

    '''

    def __init__(self, title = 'Title Not Found', poster = TEMPLATE_POSTER_URL,
                genre = 'Genre Not Found'):
        self.title = title
        self.poster = poster
        self.genre = genre
        self.imdb_id = None
        self.subtitle_features = None
        self.difficulty = []
        self.html_soup = None
        self.g_clean = [] # imdb link resulted from google searching the movie title

    def get_google_search_results(self):
        ''' Search for movie title on Google, get the first IMDB link '''
        query = self.title + ' site:imdb.com'
        url = 'https://www.google.com/search?client=ubuntu&channel=fs&q={}&ie=utf-8&oe=utf-8'.format(query)  # this is the actual query we are going to scrape
    
        url_content = requests.get(url)
        soup = BeautifulSoup(url_content.text, features="html5lib")
        a = soup.find_all('a')  # a is a list
        a[0].get('href')
        for i in a:
            k = i.get('href')
            try:
                m = re.search("(?P<url>https?://www.imdb.com/title/tt[^\s]+)", k).group()
                rul = m.split('&')[0]
                self.g_clean.append(rul)
            except:
                pass

    def get_imdb_id_from_google(self):
        ''' Get IMDB ID based on movie title via IMDB url '''
        self.get_google_search_results()
        for result in self.g_clean:
            try:
                imdb_id = result.split('tt')[-1].strip('/')
                if imdb_id:
                    self.imdb_id = str(self.imdb_id)
            except:
                print('not the link')

    def get_imdb_id_from_db(self, db):
        try:
            db_one_movie = db[db.title.str.lower() == self.title.lower()]
            self.imdb_id = str(db_one_movie.id.values[0])
        except:
            pass

    def get_imdb_id(self, db):
        try:
            self.get_imdb_id_from_db(db)
        except:
            try:
                self.get_imdb_id_from_google()
            except:
                self.imdb_id = 'not_found'

    def get_imdb_page_for_movie(self):
        url = 'https://www.imdb.com/title/tt' + self.imdb_id + '/'
        response = requests.get(url)
        self.html_soup = BeautifulSoup(response.text, 'html.parser')

    def get_movie_title(self):
        self.title = self.html_soup.title.text

    def get_link_to_movie_poster(self):
        self.poster = self.html_soup.find(class_='poster').img['src']

    def create_subtitle_features_df(self, subtitle_dir):
        """
        Extract features (see engineer_features()) from subtitles for specific movie
    
        :param: imdb_id for movie, subtitle corpus directory
        :returns: Pandas dataframe with shape (n_words, n_features)
        """
        features_list = []
        pool = Pool(processes=4)
        features_df = pd.DataFrame()
        files = os.listdir(subtitle_dir)
        list_of_text_files = []
        for file in files:
            if re.search(self.imdb_id, file):
                list_of_text_files.append(file)
        
        for episode, file in enumerate(list_of_text_files):
            filename = subtitle_dir / file
            with open(filename, 'r') as subtitles:
                texts = subtitles.read()
            sents = SENT_TOKENIZER.tokenize(texts)
            for itext in range(0, len(sents), 5):
                text_window = sents[itext:(itext+5)]
                text_window_raw = TreebankWordDetokenizer().detokenize(text_window)
                arguments = text_window_raw,  str('_'), 'tw'+ str(episode) + str(itext), 'movie'
                features_list.append(pool.apply_async(process_one_text, (arguments,)))
        for features in tqdm.tqdm(features_list):
            features_df = features_df.append(features.get(), ignore_index=True)
        self.subtitle_features = features_df

    def update_from_db(self, features_db_rec):
        ''' Update recommendation characteristics
        :param: features for the recommendation (series)
        :returns: updated Movie class with title, genre, and movie poster
        '''
        self.imdb_id = features_db_rec.id
        self.get_imdb_page_for_movie()
        self.title = features_db_rec.title
        self.genre = features_db_rec.genre
        self.get_link_to_movie_poster()

    def show_difficulty(subtitle_dir, model_dir, model = 'regression'):
        ''' Predict a language proficiency label for every 5 sentences in a movie to infer
        text difficulty for language learners
        :param: movie_title (title of the movie, str), subtitle_dir (directory with subtitle
        database, model_dir (directory with movielingo_model.sav - pickled model), model 
        (classifier or regression)
        :returns: results (list of proficiency labels with corresponding ratio of text:
        e.g., [["A1", 56], ["B2", 44]]), URL for movie poster, plotly pie chart based on results
        ''' 
        loaded_model_name = model_dir / 'movielingo_model.sav'
        loaded_model = pickle.load(open(loaded_model_name, 'rb'))
        text_preds = []
        for text_id in self.subtitle_features.text_id.unique():
            text_slice = self.subtitle_features[self.subtitle_features.text_id == text_id]
            text_slice = text_slice.drop(columns = ['text_id','L2_proficiency']).reset_index(drop=True)
            text_pred = loaded_model.predict(text_slice)
            text_preds.append(text_pred)
        if model == 'regression':
            prof_labels = [toeic2cefr(float(x)) for x in text_preds[0].tolist()]
        else:
            prof_labels = text_preds
        levels = Counter(prof_labels).keys()
        classified_as = list(Counter(prof_labels).values())
        n_windows = sum(Counter(prof_labels).values())
        self.difficulty = []
        for level, label_count in zip(levels, classified_as):
            self.difficulty.append([level, round(100*label_count/n_windows,2)])

if __name__ == '__main__':
    movie = Movie('Casablanca')