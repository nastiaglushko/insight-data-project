from flask import Flask

app = Flask(__name__, template_folder = 'templates')

__all__ = ['text_utils', 'single_text_processor',
			'batch_text_processing_multi', 'movielingo_recommend',
			'movielingo_classify', 'movie', 'get_nearest_neighbours',
			'run_models_sklearn', 'config', 'modelling', 'movielingo_model',
			'test_config', 'test_custom_funcs', 'parser_patterns']

from movielingo import batch_text_processing_multi
from movielingo import config
from movielingo import single_text_processor
from movielingo import text_utils
from movielingo import modelling
from movielingo import movielingo_recommend
from movielingo import movielingo_classify
from movielingo import movie
from movielingo import get_nearest_neighbours
from movielingo import run_models_sklearn




