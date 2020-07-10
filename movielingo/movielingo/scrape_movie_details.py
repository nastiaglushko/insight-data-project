import json
import os
import re
import pandas as pd

from requests import get
from bs4 import BeautifulSoup
import time
import tqdm

from requests import get
from bs4 import BeautifulSoup

from movielingo.config import subtitle_dir
from movielingo.config import processed_data_dir

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
	for imdb_id in tqdm.tqdm(imdb_ids_list):
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

if __name__ == '__main__':
	files = os.listdir(subtitle_dir)
	ids = []
	for file in files:
	    filename = subtitle_dir / file
	    imdb_id = re.search("(\d{7})", file).group()
	    ids.append(imdb_id)
	unique_ids = list(set(ids))
	filename = processed_data_dir / 'movie_characteristics.txt'
	scrape_movie_details(unique_ids[9000:], filename)
	imdb = pd.read_csv(filename,
                   sep = '\t',
                   names = ['id', 'title', 'genre', 'movie_or_show', 'keywords', 'rating'],
                   dtype = {'id': str, 'rating': float})
	imdb.loc[:,'title'] = imdb.title.str.split(' \(.{4,40}\) - IMDb', expand=True)[0].str.strip()
	imdb.to_csv(processed_data_dir / 'imdb_title_and_id_matches.csv', index = False)