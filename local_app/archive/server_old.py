import sys
sys.path.append("../movielingo/")
from movielingo.movie_info_output import show_difficulty, get_result
from movielingo.modelling import FeatureRecorder, FeatureSelector, ClfSwitcher
from movielingo.config import subtitle_dir, model_dir
from flask import Flask, render_template, request
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
import os
import io

# Create the application object
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('main.html')

@app.route('/', methods=["GET", "POST"]) # we are now using these methods to get user input
def dynamic_page():
    if request.method == 'POST':
        movie_title = str(request.form['movie_title'])
        l2_level = str(request.form['l2_level'])
        bar_plot_data, imdb_poster, plot = show_difficulty(movie_title, subtitle_dir, model_dir, model = 'regression')
        result = get_result(bar_plot_data, l2_level)
        return render_template('final.html', result=result, movie_title=movie_title,imdb_poster = imdb_poster, plot = plot)

if __name__ == '__main__':
    app.run(debug=True)  # will run locally
