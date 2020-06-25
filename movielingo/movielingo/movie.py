import requests
from bs4 import BeautifulSoup
import pandas as pd
from movielingo.batch_text_processing_multi import process_one_text
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import tqdm
import os
from multiprocessing import Pool
from movielingo.config import subtitle_dir, processed_data_dir
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktLanguageVars

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '•', '...')

SENT_TOKENIZER = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
TEMPLATE_POSTER_URL = 'https://s.studiobinder.com/wp-content/uploads/2017/12/Movie-Poster-Template-Light-With-Image.jpg?x81279'
IMDB = pd.read_csv(processed_data_dir /'movie_details_db.csv', dtype = {'id': str})

class Movie():
    ''' A movie in the Movielingo app

    Main attributes:
    - title (str)
    - IMDB ID (str)
    - link to movie poster (str)
    - subtitle features (pandas df with NLP features)

    '''

    def __init__(self, title = 'Title Not Found', poster = TEMPLATE_POSTER_URL,
                genre = 'Genre Not Found'):
        self.title = title
        self.poster = poster
        self.genre = genre
        self.imdb_id = None
        self.subtitle_features = None
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

        self.imdb_id = features_db_rec.id
        self.get_imdb_page_for_movie()
        self.title = features_db_rec.title
        self.genre = features_db_rec.genre
        self.get_link_to_movie_poster()

if __name__ == '__main__':
    movie = Movie('Casablanca')
    movie.get_imdb_id(IMDB)
    movie.get_imdb_page_for_movie()
    movie.get_link_to_movie_poster()
    movie.create_subtitle_features_df(subtitle_dir)
    movie.update_from_db(IMDB)