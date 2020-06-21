import json
import numpy as np
import os
import pandas as pd
import re
import time
import tqdm

from requests import get
from bs4 import BeautifulSoup

from movielingo.batch_text_processing_multi import create_df_from_subtitles
from movielingo.config import subtitle_dir, model_dir

from sklearn.neighbours import NearestNeighbours
import pickle


def read_imdb_list(imdb_soup):
	''' Get titles and IMDB ids of movies from an IMDB top-XXX list
	:param: BeautifulSoup object (IMDB list)
	:returns: IMDB IDs (str) and titles (str)
	'''
	entry = soup.find_all('td', {"class": 'titleColumn'})
	a = []
	for movie in entry:
	    a.append(movie.find_all(lambda tag: tag.name == "a" and not tag.find_all("td")))
	
	top_ids = []
	top_titles = []
	for i in range(len(a)):
	    top_ids.append(a[i][0]['href'].split('/title/tt')[1].split('/')[0])
	    top_titles.append(a[i][0].text)
	return top_ids, top_titles

def get_available_subtitles(directory):
	''' Get IDs of movies, for which subtitles are in our database
	:param: directory with subtitles formatted as in http://ghpaetzold.github.io/subimdb/
	:returns: list of IMDB IDs for available subtitles
	'''
	files = os.listdir(directory)
	db_ids = []
	for file in files:
	    filename = subtitle_dir / file
	    imdb_id = re.search("(\d{7})", file).group()
	    db_ids.append(imdb_id)
	return list(set(db_ids))

def get_top_movies_with_subtitles(top_ids, top_titles, subtitle_ids):
	''' Get list of top movies, for which subtitles are available
	:param: lists of top movie IMDB IDs, titles, and list of IDs for movies in subtitle db
	:returns: list of tuples (movie_id, title)
	'''
	good_ids_with_subtitles = [ID for ID in top_ids if ID in subtitle_ids]
	good_titles_with_subtitles = [t for ID, t in zip(top_ids, top_titles) if ID in subtitle_ids]
	return good_ids_with_subtitles, good_titles_with_subtitles

def get_movie_characteristics(imdb_id):
	''' Get movie characteristics from IMDB
	:param: IMDB ID
	:returns: movie details as strings
	'''
	url = 'https://www.imdb.com/title/tt' + imdb_id + '/'
	response = get(url)
	html_soup = BeautifulSoup(response.text, 'html.parser')
	title = html_soup.title.text
	description = html_soup.find('script', type="application/ld+json").contents[0]
	description = json.loads(description)
	genre = description['genre']
	movie_or_tv = description['@type']
	keywords = description['keywords']
	if type(keywords) == list:
	    keywords = ','.join(keywords)
	if type(genre) == list:
	    genre = ','.join(genre)
	rating = description['aggregateRating']['ratingValue']
	return title, genre, movie_or_tv, keywords, rating

def scrape_movie_details(imdb_ids_list, filename):
	''' Write movie characteristics into a text file
	:param: list of IMDB ids, filename
	:returns: None (writes to file)
	'''
	for imdb_id in tqdm.tqdm(ids_in_db):
		try:
			title, genre, movie_or_tv, keywords, rating = get_movie_characteristics(imdb_id)
			time.sleep(5)
			with open(filename, 'a') as movie_characteristics_file:
			    movie_characteristics_file.write(imdb_id + '\t' + 
			                              title + '\t' + 
			                              genre + '\t' +
			                              movie_or_tv + '\t' +
			                              keywords + '\t' +
			                              rating +
			                              '\n')
		except:
			pass

def extract_features_from_subtitles(imdb_ids, titles):
	''' Extract features from subtitles
	:param: list of ids present in the subtitle database
	:returns: pandas df (feature extracted for every 5 sentences), df_summary (one line per movie)
	'''
	for movie in range(len(imdb_ids)):
	    df_movie = create_df_from_subtitles(imdb_ids[movie], subtitle_dir)
	    df_movie['title'] = titles[movie]
	    if movie == 0:
	    	df = df_movie
	    else:
	    	df = df.append(df_movie, ignore_index = True)
	df_summary = df.groupby('title').mean()
	return df, df_summary

def save_NN_fits(samples, directory):
	''' Fit and save nearest neighbour models for 1:10 neighbours
	:param: samples (numpy array)
	:returns: None (writes files to directory)
	'''
	for n in range(10):
		neigh = NearestNeighbors(n_neighbors=n, metric="cosine")
		neigh.fit(X)
		filename = model_dir / str(n) + '_neighbours_model.sav'
		pickle.dump(neigh, open(filename, 'wb'))

if __name__ == '__main__':
	response = get('https://www.imdb.com/chart/top-english-movies')
	soup = BeautifulSoup(response.text, 'html.parser')
	top_ids, top_titles = read_imdb_list(soup)
	subtitle_ids = get_available_subtitles(subtitle_dir)
	ids, titles = get_top_movies_with_subtitles(top_ids, top_titles, subtitle_ids)
	df, df_summary = extract_features_from_subtitles(ids, titles)
	y = df_summary.index
	X = df_summary.values
	save_NN_fits(X, model_dir)

