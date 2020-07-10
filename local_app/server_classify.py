import sys
sys.path.append("../movielingo/")
from movielingo.movielingo_classify import get_result, plot_subtitle_difficulty
from movielingo.modelling import FeatureRecorder, FeatureSelector, ClfSwitcher
from movielingo.config import subtitle_dir, model_dir, processed_data_dir
from movielingo.movie import Movie
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
    return render_template('main_classify.html')

@app.route('/', methods=["GET", "POST"]) # we are now using these methods to get user input
def dynamic_page():
    if request.method == 'POST':
        movie_title = str(request.form['movie_title'])
        l2_level = str(request.form['l2_level'])
        user_movie = Movie(title = movie_title)
        user_movie.get_imdb_id(movie_db)
        user_movie.get_imdb_page_for_movie()
        user_movie.get_link_to_movie_poster()
        user_movie.create_subtitle_features_df(subtitle_dir)
        user_movie.show_difficulty()
        plot = plot_subtitle_difficulty(user_movie)
        result = get_result(user_movie, l2_level)
        print(result)
        return render_template('final_classify.html', result=result, movie_title=movie_title,imdb_poster = user_movie.poster, plot = plot)

if __name__ == '__main__':
    app.run(debug=True)  # will run locally
