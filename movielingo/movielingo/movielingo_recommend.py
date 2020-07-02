import pandas as pd

from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from movielingo.modelling import FeatureRecorder, FeatureSelector, ClfSwitcher
from movielingo.movie import Movie
from pathlib import Path
import pickle

#IMDB = pd.read_csv(processed_data_dir / 'imdb_title_and_id_matches.csv', dtype = {'id': str})
IMDB = pd.read_csv(processed_data_dir /'movie_details_db.csv', dtype = {'id': str})

def recommend_movies(movie_title_from_user, movie_db, pref_genre):
    ''' Get movie recommendations based on inputted movie
    :param: movie title (str) for the movie user enjoyed
    :returns: movie, first recommendation, second recommendation (class Movie)
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

if __name__ == '__main__':
    user_movie, recommend1, recommend2 = recommend_movies('Casablanca', IMDB, 'Drama')
    print(recommend1.title)
    print(recommend2.title)
