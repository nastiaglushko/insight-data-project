import sys
sys.path.append("../movielingo/")
#from movielingo import app
from movielingo.movie_info_output import recommend_movies
from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from flask import Flask, render_template, request
from flask import Response
import pandas as pd
import os
import io

movie_db = pd.read_csv(processed_data_dir /'movie_details_db.csv', dtype = {'id': str})

#Create the application object
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('main_recom.html')

@app.route('/', methods=["GET", "POST"]) # we are now using these methods to get user input
def dynamic_page():
    if request.method == 'POST':
        movie_title = str(request.form['movie_title'])
        preferred_genre = str(request.form['desired_genre']) # can change this to genre
        user_movie, recommend1, recommend2 = recommend_movies(movie_title, movie_db, preferred_genre)
        return render_template('final_recom.html', original_movie=user_movie.title, imdb_poster = user_movie.poster,
        											movie_title1=recommend1.title, imdb_poster1 = recommend1.poster, genres1 = recommend1.genre,
        											movie_title2=recommend2.title, imdb_poster2 = recommend2.poster, genres2 = recommend2.genre)

if __name__ == '__main__':
    app.run(debug=True)  # will run locally
