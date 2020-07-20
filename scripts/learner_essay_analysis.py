#### ----- English learner's essay analysis pipeline ----- ####

'''
1. Engineer features from text files
2. Model data using sklearn
3. Model data using keras

Note: this script outlines the order in which the different analysis steps
have been done. These steps (1,2,3) have been run separately though,
which can be done by running 1. batch_processing_multi.py, 2. run_models_sklearn.py,
3. run_models_keras.py to allow for more flexibility (e.g., in selecting
hyperparameters for ML models etc.)

Another note: to get syntactic (dependency) features,
run corenlpserver.py starting the StanfordNLP server prior to create_df_from_texts

'''

import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from movielingo.batch_text_processing_multi import create_df_from_texts
from movielingo.config import processed_data_dir
from movielingo.modelling_utils import *
from movielingo.run_models_keras import *

from movielingo.run_models_sklearn import get_learners_df
from movielingo.run_models_sklearn import split_data
from movielingo.run_models_sklearn import split_data
from movielingo.run_models_sklearn import build_pipeline
from movielingo.run_models_sklearn import fit_pipeline
from movielingo.run_models_sklearn import run_grid_search_CV
from movielingo.run_models_sklearn import save_pickled_model

import keras
import pickle

# 1. Engineer features from text files

input_filename = os.path.join(processed_data_dir, '01_gachon_processed.csv')

features_df, failures = create_df_from_texts(input_filename)

features_filename = os.path.join(processed_data_dir, '02_gachon_features.csv')
features_df.to_csv(features_filename, index = False)

# 2. Model data using sklearn - regression

df = get_learners_df(features_filename, prof_levels = ['Low', 'A1', 'A2', 'B1', 'B2+'])
X_train, X_test, y_train, y_test = split_data(df)
benchmark_pipeline = build_pipeline(ElasticNet())
benchmark_pipeline = fit_pipeline(benchmark_pipeline, X_train, y_train, 'benchmark_model_results.txt')
best_model = run_grid_search_CV(benchmark_pipeline, X_train, y_train, 'final_model_results.txt')
save_pickled_model(best_model, 'movielingo_model.sav')

# 3. Model data using keras - classification

df = get_learners_df(features_filename, prof_levels = [0,0,0,1,1])
X_train, X_test, y_train, y_test = split_data(df, stratify_by = 'cefr', y_column = 'cefr')
prep_pipeline = get_preprocessing_pipeline()
X_tr = prep_pipeline.fit_transform(X_train)
y_tr = keras.utils.to_categorical(y_train.values, num_classes= 2)
X_te = prep_pipeline.transform(X_test)
y_te = keras.utils.to_categorical(y_test.values, num_classes= 2)
nn = build_nn()
nn.compile(optimizer=keras.optimizers.Adagrad(learning_rate=1e-6),
           loss=keras.losses.CategoricalCrossentropy(),
           metrics=[keras.metrics.CategoricalAccuracy()])
nn.fit(X_tr, y_tr,
       batch_size=128,
       epochs=2500)
test_loss, test_acc = nn.evaluate(X_te, y_te)
print('test_acc: ', test_acc)
y_pred = nn.predict(X_te)
matrix = confusion_matrix(y_te.argmax(axis=1), y_pred.argmax(axis=1), normalize = 'true')

# plot confusion matrix
sns.set()
conf = sns.heatmap(matrix, vmin=0, vmax=1, annot=True, linewidths=1, cmap="YlGnBu")
conf.set(xticklabels=['Low', 'High'], yticklabels=['Low','High'], ylabel = 'True', xlabel = 'Predicted');
plt.savefig('confusion_matrix.pdf')
save_pickled_model(nn, 'movielingo_nn_model.sav')