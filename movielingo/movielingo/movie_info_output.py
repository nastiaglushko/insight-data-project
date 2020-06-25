from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from movielingo.modelling import FeatureRecorder, FeatureSelector, ClfSwitcher
from movielingo.movie import Movie
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
        
def show_difficulty(movie_title, subtitle_dir, model_dir, model = 'regression'):
    ''' [Used for version 1 of Movielingo app; now out of use]
    Predict a language proficiency label for every 5 sentences in a movie to infer
    text difficulty for language learners
    :param: movie_title (title of the movie, str), subtitle_dir (directory with subtitle
    database, model_dir (directory with movielingo_model.sav - pickled model), model 
    (classifier or regression)
    :returns: results (list of proficiency labels with corresponding ratio of text:
    e.g., [["A1", 56], ["B2", 44]]), URL for movie poster, plotly pie chart based on results
    ''' 
    imdb_id = get_imdb_id(movie_title, IMDB)
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
    
def plot_subtitle_difficulty(difficulty_results, movie_title):
    ''' [Used for version 1 of Movielingo app; now out of use]
    :param: list of proficiency labels with corresponding ratio of text:
    e.g., [["A1", 56], ["B2", 44]], movie_title (str)
    :returns: plotly pie chart in html format
    '''
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

def get_result(results, l2_level):
    ''' [Used for version 1 of Movielingo app; now out of use]
    Based on language complexity composite score + heuristics, produces
    actionable feedback for language learner re: whether the movie is good for them
    :param: list of proficiency labels with corresponding ratio of text:
    e.g., [["A1", 56], ["B2", 44]], language proficiency level ('BegInter' / 'UpperInterAdv')
    :results: " (movie X) is just right for you!" etc.
    '''
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

def recommend_movies(movie_title_from_user, movie_db, pref_genre):
    ''' Get movie recommendations based on inputted movie
    :param: movie title (str) for the movie user enjoyed
    :returns: movie titles (list), links to movie posters (list), genre (list)
    '''
    user_movie = Movie(title = movie_title_from_user)
    user_movie.get_imdb_id(IMDB)
    recommend1 = Movie()
    recommend2 = Movie()
    if user_movie.imdb_id != 'not_found':
        user_movie.get_imdb_page_for_movie()
        user_movie.get_link_to_movie_poster()
        user_movie.create_subtitle_features_df(subtitle_dir)
        X = user_movie.subtitle_features.mean(axis = 0).values.reshape(1, -1)
        model_name = model_dir / '5_neighbours_model.sav'
        neigh = pickle.load(open(model_name, 'rb'))
        _, neigh_ind = neigh.kneighbors(X)
        neigh_ind = neigh_ind.flatten()
        genres = list(movie_db.iloc[neigh_ind]['genre'].values)
        pref_genre_inds = neigh_ind[[i for i, genre in enumerate(genres) if pref_genre in genre]]
        n_right_genre = len(pref_genre_inds)
        if n_right_genre > 0:
            features_rec1 = movie_db.iloc[pref_genre_inds[0]]
            recommend1.update_from_db(features_rec1)
            if n_right_genre > 1:
                features_rec2 = movie_db.iloc[pref_genre_inds[1]]
                recommend2.update_from_db(features_rec2)
            else:
                features_rec2 = movie_db.iloc[neigh_ind[0]]
                recommend2.update_from_db(features_rec2)
        else:
            features_rec1 = movie_db.iloc[neigh_ind[0]]
            recommend1.update_from_db(features_rec1)
            features_rec2 = movie_db.iloc[neigh_ind[1]]
            recommend2.update_from_db(features_rec2)

    return user_movie, recommend1, recommend2