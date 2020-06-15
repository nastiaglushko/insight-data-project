from flask import Flask
app = Flask(__name__, template_folder = 'templates')

__all__ = ["text_utils", 'single_text_processor', 'batch_text_processing_multi', 'movie_info_output', 'config', 'modelling', 'movielingo_model', 'test_config', 'test_custom_funcs']