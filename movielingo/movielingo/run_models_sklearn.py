import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

import pickle

sys.path.append("../movielingo/")
from movielingo.modelling import *
from movielingo.config import processed_data_dir

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def get_learners_df(filename = 'gachon_features_1706_full.csv'):
    ''' Process dataframe for sklearn '''
    df = pd.read_csv(processed_data_dir / filename, index_col = 0)
    df['L2_proficiency'] = df['L2_proficiency'].astype(float)
    df['cefr'] = df.apply(toeic2cefr, axis = 1)
    df['L2_proficiency'] = df['L2_proficiency'] / 1000
    df = df[df.cefr != 'Low'].reset_index(drop=True)
    return df

def split_data(df):
    ''' Split data into training and testing sets '''

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=19)
    for train_index, test_index in split.split(df, df.cefr):
        X_train = df.loc[train_index]
        X_test = df.loc[test_index]

        X_train.drop(columns=['cefr'], inplace = True)
        X_test.drop(columns=['cefr'], inplace = True)
        y_train = X_train.L2_proficiency
        X_train = X_train.drop('L2_proficiency', axis = 1)
        y_test = X_test.L2_proficiency
        X_test = X_test.drop('L2_proficiency', axis = 1)
        return X_train, X_test, y_train, y_test

def build_pipeline(estimator = LinearRegression()):
    ''' Build ML pipeline in sklearn '''
    num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'constant')), # fill_value default = 0
    ('std_scaler', StandardScaler())
    ])

    pipeline = Pipeline([
    ('prep_data', num_pipeline),
    ('add_interactions', PolynomialFeatures(interaction_only = True)),
    ('feature_selection', FeatureSelector()), # None by default
    ('clf', ClfSwitcher(estimator))
    ])
    return pipeline

def fit_pipeline(pipeline, X_train, y_train, results_file):
    ''' Fit ML pipeline, run cross-validation, write scores to file '''
    pipeline.fit(X_train, y_train);
    pipeline_scores2file(pipeline, X_train, y_train, results_file)
    return pipeline

def pipeline_scores2file(pipeline, X, y, filename):
    predictions = pipeline.predict(X)
    r2  = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    cross_val_scores = cross_val_score(pipeline, X, y,
       scoring = 'neg_mean_squared_error', cv = 10)
    results = open(filename, 'a')
    results.write('R2:' + str(r2) + '\nRMSE' + str(np.sqrt(mse)) + '\n')
    results.close()
    write_pipeline_scores(cross_val_scores, filename)

def get_r2_from_model(pipeline, X_train, y_train, X_test, y_test, filename):
    predictions_train = pipeline.predict(X_train)
    predictions_test = pipeline.predict(X_test)
    r2_train  = r2_score(y_train, predictions_train)
    r2_test  = r2_score(y_test, predictions_test)
    results = open(filename, 'a')
    results.write('R2 for training set:' + str(r2_train) + '\n' + 'R2 for test set' + str(r2_test) + '\n')
    results.close()

def run_grid_search_CV(pipeline, X_train, y_train, results_filename):
    parameters = [
    {
    'add_interactions__degree': [1, 2],
    'add_interactions__interaction_only': [True],
    'feature_selection__estimator': [SelectFromModel(LinearRegression())],
    'feature_selection__estimator__threshold': ['0.5*mean', '0.25*mean'],
    'clf__estimator': [RandomForestRegressor()],
    'clf__estimator__min_samples_split': [20, 30, 40, 50],
    'clf__estimator__n_estimators': [100, 200, 300, 400],
    'clf__estimator__max_features': ['auto', 'sqrt'],
    'clf__estimator__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'clf__estimator__bootstrap': [True, False]
    # 'clf__estimator__loss': ['ls', 'lad', 'huber', 'quantile'],
    # 'clf__estimator__learning_rate': [0.0001, 0.001, 0.01]
    }
    ]

    grid_search = RandomizedSearchCV(pipeline, parameters, cv=5,
      scoring = 'neg_mean_squared_error', verbose = 2,
      return_train_score = True, n_iter=25, n_jobs = -1)

    grid_search_results = grid_search.fit(X_train, y_train)

    best_model = grid_search_results.best_estimator_

    write_best_grid_search_score(grid_search_results, results_filename)
    get_r2_from_model(best_model, X_train, y_train, X_test, y_test, results_filename)

    return best_model

def save_pickled_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
   df = get_learners_df()
   X_train, X_test, y_train, y_test = split_data(df)
   benchmark_pipeline = build_pipeline(ElasticNet())
   benchmark_pipeline = fit_pipeline(benchmark_pipeline, X_train, y_train, 'benchmark_model_results.txt')
   best_model = run_grid_search_CV(benchmark_pipeline, X_train, y_train, 'final_model_results.txt')
   save_pickled_model(best_model, 'movielingo_model.sav')
