from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from movielingo.batch_text_processing_multi import create_df_from_subtitles, engineer_features
from movielingo.modelling import FeatureRecorder, FeatureSelector, ClfSwitcher
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
import requests
import re
import pandas as pd
import pickle
from pathlib import Path
from collections import Counter
import numpy as np

#IMDB = pd.read_csv(processed_data_dir / 'imdb_title_and_id_matches.csv', dtype = {'id': str})
IMDB = pd.read_csv(processed_data_dir /'movie_details_db.csv', dtype = {'id': str})

def get_google_search_results(movie): # googled blocked my IP, switching to use get_imdb_id_from_db
    query = movie + ' site:imdb.com'
    g_clean = []  # this is the list we store the search results
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
           g_clean.append(rul)
       except:
           pass
    return g_clean

def get_imdb_id(movie):
    g_clean = get_google_search_results(movie)
    for result in g_clean:
        try:
            imdb_id = result.split('tt')[-1].strip('/')
            if imdb_id:
                return imdb_id
        except:
            print('not the link')

def toeic2cefr(x):
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
        
def show_difficulty(movie_title, subtitle_dir, model_dir, model = 'regression'):
    imdb_id = get_imdb_id_from_db(movie_title)
    html_soup = get_imdb_page_for_movie(imdb_id)
    movie_poster_link = get_link_to_movie_poster(html_soup)
    movie_title_correct = get_movie_title(html_soup)
    df = create_df_from_subtitles(imdb_id, subtitle_dir)
    loaded_model_name = model_dir / 'movielingo_model.sav'
    loaded_model = pickle.load(open(loaded_model_name, 'rb'))
    text_preds = []
    for text_id in df.text_id.unique():
        text_slice = df[df.text_id == text_id]
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
    results = []
    for level, label_count in zip(levels, classified_as):
        results.append([level, round(100*label_count/n_windows,2)])
    plot = plot_subtitle_difficulty(results, movie_title_correct)
    return results, movie_poster_link, plot
    
def get_imdb_page_for_movie(imdb_id): # for compiling a list of movies in the subtitle database
    url = 'https://www.imdb.com/title/tt' + imdb_id + '/'
    response = requests.get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    return html_soup

def get_movie_title(imdb_page_html):
    return imdb_page_html.title.text

def get_link_to_movie_poster(imdb_page_html):
    return imdb_page_html.find(class_='poster').img['src']
    
def plot_subtitle_difficulty(difficulty_results, movie_title):
    labels = []
    vals = []
    for i in range(len(difficulty_results)):
        labels.append(difficulty_results[i][0])
        vals.append(difficulty_results[i][1])
    fig = px.pie(labels=labels, values=vals, names = labels, hole=.3, color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    }, autosize = False, width = 400, height = 400)
    fig.update_layout({'margin': dict(l=0, r=0, t=50, b=50)},
    				  {'font': {'family': 'Old Standard TT', 'size': 14, 'color': 'white'}},
                      showlegend=False)
    div = fig.to_html(full_html = True)
    return div

def get_imdb_id_from_db(movie_title):
    try:
        IMDB_one_movie = IMDB[IMDB.title.str.lower() == movie_title.lower()]
        imdb_id = IMDB_one_movie.id.values[0]
    except:
        imdb_id = 'not_found'
    return str(imdb_id)

def get_result(results, l2_level):
    labels = []
    vals = []
    for i in range(len(results)):
        labels.append(results[i][0])
        vals.append(results[i][1])
    b1_can_understand = 0
    if 'A2' in labels:
        i = labels.index('A2')
        b1_can_understand += vals[i]
    if 'B1' in labels:
        i = labels.index('B1')
        b1_can_understand += vals[i]
    if l2_level == 'BegInter':
        if b1_can_understand >= 75:
            result = 'is just right for you!'
        elif b1_can_understand >= 50:
            result = 'might be a bit too difficult for you!'
        else:
            'is probably too difficult for you.'
    if l2_level == 'UpperInterAdv':
        if b1_can_understand >= 75:
            result = 'is almost too easy for you!'
        else:
            result = 'is just right for you!'
    return result

def recommend_movies(movie_title_from_user, movie_db, preferred_genre):
    ''' Get movie recommendations based on inputted movie
    :param: movie title (str) for the movie user enjoyed
    :returns: movie titles (list), links to movie posters (list), genre (list)
    '''
    imdb_id = get_imdb_id_from_db(movie_title_from_user)
    if imdb_id != 'not_found':
        html_soup = get_imdb_page_for_movie(imdb_id)
        orig_movie_poster_link = get_link_to_movie_poster(html_soup)
        df = create_df_from_subtitles(imdb_id, subtitle_dir)
        df['title'] = movie_title_from_user
        df_summary = df.groupby('title').mean()
        model_name = model_dir / '5_neighbours_model.sav'
        neigh = pickle.load(open(model_name, 'rb'))
        X = df_summary[df_summary.index == movie_title_from_user].values
        _, neigh_ind = neigh.kneighbors(X)
        neigh_ind = neigh_ind[0]
        titles = list(movie_db.iloc[neigh_ind]['title'].values)
        ids = movie_db.iloc[neigh_ind]['id'].values
        genres = list(movie_db.iloc[neigh_ind]['genre'].values)
        selected_by_genre = [i for i, genre in enumerate(genres) if preferred_genre in genre]
        posters = []
        for ID in ids:
            html_soup = get_imdb_page_for_movie(ID)
            movie_poster_link = get_link_to_movie_poster(html_soup)
            posters.append(movie_poster_link)
        if len(selected_by_genre) > 1:
            titles = [titles[i] for i in selected_by_genre]
            posters = [posters[i] for i in selected_by_genre]
            genres = [genres[i] for i in selected_by_genre]
        else:
            titles = [titles[i] for i in selected_by_genre] + titles
            posters = [posters[i] for i in selected_by_genre] + posters
            genres = [genres[i] for i in selected_by_genre] + genres
        posters.append(orig_movie_poster_link)
        titles.append(movie_title_from_user)
    else:
        titles = ["Not found", "Not found", "Not found", "Not found", "Not found", "Sorry, Movielingo doesn't have subtitles for " + movie_title_from_user]
        poster_template = 'https://s.studiobinder.com/wp-content/uploads/2017/12/Movie-Poster-Template-Light-With-Image.jpg?x81279'
        posters = [poster_template,poster_template,poster_template,poster_template,poster_template,poster_template]
        genres = ['', '', '']
    return titles, posters, genres