import numpy as np
import os
import pandas as pd
import re

from requests import get
from bs4 import BeautifulSoup

from movielingo.movie import Movie
from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from movielingo.scrape_movie_details import scrape_movie_details

from sklearn.neighbors import NearestNeighbors
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

def extract_features_from_subtitles(imdb_ids, titles):
	''' Extract features from subtitles
	:param: list of ids present in the subtitle database, titles of these movies
	:returns: pandas df (feature extracted for every 5 sentences), df_summary (one line per movie)
	'''
	for movie in range(len(imdb_ids)):
		one_movie = Movie()
		one_movie.imdb_id = imdb_ids[movie]
		one_movie.create_subtitle_features_df(subtitle_dir)
		df_movie = one_movie.subtitle_features
		df_movie['title'] = titles[movie]
		if movie == 0:
			df = df_movie
			df.to_csv(processed_data_dir / 'movie_features_temp.csv', index=False)
		else:
			df = df.append(df_movie, ignore_index = True)
			df.to_csv(processed_data_dir / 'movie_features_temp.csv', index=False)
	df_summary = df.groupby('title').mean()
	df_summary.to_csv(processed_data_dir / 'movie_features.csv', index=False)
	print(df_summary)
	return df, df_summary

def save_NN_fits(samples, directory):
	''' Fit and save nearest neighbour models for 1:10 neighbours
	:param: samples (numpy array)
	:returns: None (writes files to directory)
	'''
	for n in range(1,10):
		neigh = NearestNeighbors(n_neighbors=n, metric="cosine")
		neigh.fit(X)
		model_name = str(n) + '_neighbours_model.sav'
		pickle.dump(neigh, open(model_name, 'wb'))

if __name__ == '__main__':
	response = get('https://www.imdb.com/chart/top-english-movies')
	soup = BeautifulSoup(response.text, 'html.parser')
	top_ids, top_titles = read_imdb_list(soup)
	subtitle_ids = get_available_subtitles(subtitle_dir)
	ids, titles = get_top_movies_with_subtitles(top_ids, top_titles, subtitle_ids)
	df, df_summary = extract_features_from_subtitles(ids[-1], titles[-1])
	df_summary.to_csv('movie_features.csv', index=False)
	y = df_summary.index
	X = df_summary.values
	save_NN_fits(X, model_dir)
	#filename = processed_data_dir / 'top250_movie_characteristics.txt'
	#scrape_movie_details(ids, filename)

