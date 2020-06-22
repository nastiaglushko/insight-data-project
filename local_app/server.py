import sys
sys.path.append("../movielingo/")
from movielingo.movie_info_output import recommend_movies
from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from flask import Flask, render_template, request
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
import pandas as pd
import os
import io

movie_db = pd.read_csv(processed_data_dir /'movie_details_db.csv', dtype = {'id': str})

# Create the application object
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('main_recom.html')

@app.route('/', methods=["GET", "POST"]) # we are now using these methods to get user input
def dynamic_page():
    if request.method == 'POST':
        movie_title = str(request.form['movie_title'])
        # l2_level = str(request.form['l2_level']) # can change this to genre
        titles, posters, genres = recommend_movies(movie_title, movie_db)
        return render_template('final_recom.html', original_movie=titles[-1], imdb_poster = posters[-1],
        											movie_title1=titles[0], imdb_poster1 = posters[0], genres1 = genres[0],
        											movie_title2=titles[1], imdb_poster2 = posters[1], genres2 = genres[1])

if __name__ == '__main__':
    app.run(debug=True)  # will run locally
