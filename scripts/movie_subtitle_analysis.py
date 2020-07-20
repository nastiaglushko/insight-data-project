#### ----- Subtitle analysis pipeline ----- ####

'''
1. Engineer features from subtitles of top-rated IMDB movies
2. Scrape movie characteristics (genre, rating etc.)
3. Build nearest neighbours model

Note: this script outlines the order in which the different analysis steps
have been done. These steps (1,2,3) have been run separately though,
which can be done by running get_nearest_neighbours.py and scrape_movie_details.py
to allow for more flexibility. + Scraping from IMDB integrates a wait of 5 seconds
between each movie, so it takes a while.

Another note: to get syntactic (dependency) features,
run corenlpserver.py starting the StanfordNLP server prior to extract_features_from_subtitles
'''

import numpy as np
import os
import pandas as pd
import re

from requests import get
from bs4 import BeautifulSoup

from sklearn.neighbors import NearestNeighbors
import pickle

from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from movielingo.get_nearest_neighbours import *
from movielingo.movie import Movie
from movielingo.scrape_movie_details import scrape_movie_details

# 1. Engineer features from subtitles of top-rated IMDB movies

response = get('https://www.imdb.com/chart/top-english-movies')
soup = BeautifulSoup(response.text, 'html.parser')
top_ids, top_titles = read_imdb_list(soup)
subtitle_ids = get_available_subtitles(subtitle_dir)
ids, titles = get_top_movies_with_subtitles(top_ids, top_titles, subtitle_ids)
df, df_summary = extract_features_from_subtitles(ids, titles)

# 2. Scrape movie characteristics (genre, rating etc.) for movies in subtitle database

# for top IMDB movies
filename = processed_data_dir / 'top250_movie_characteristics.txt'
scrape_movie_details(ids, filename)

# for movies in subtitle database
files = os.listdir(subtitle_dir)
ids = []
for file in files:
    filename = subtitle_dir / file
    imdb_id = re.search("(\d{7})", file).group()
    ids.append(imdb_id)
unique_ids = list(set(ids))
filename = processed_data_dir / 'movie_characteristics.txt'
scrape_movie_details(unique_ids, filename)
imdb = pd.read_csv(filename,
                sep = '\t',
                names = ['id', 'title', 'genre', 'movie_or_show', 'keywords', 'rating'],
                dtype = {'id': str, 'rating': float})
imdb.loc[:,'title'] = imdb.title.str.split(' \(.{4,40}\) - IMDb', expand=True)[0].str.strip()
imdb.to_csv(processed_data_dir / 'imdb_title_and_id_matches.csv', index = False)

# 3. Build nearest neighbours model

y = df_summary.index
X = df_summary.values
save_NN_fits(X, model_dir)

