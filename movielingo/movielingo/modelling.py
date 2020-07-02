import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

class FeatureRecorder(BaseEstimator, TransformerMixin):

    def __init__(self, estimator = None):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X = X
        return X

    def get_feature_names(self):
        return self.X.columns.tolist()

class FeatureSelector(MetaEstimatorMixin, BaseEstimator):

    def __init__(self, estimator = None):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        if self.estimator:
            self.estimator.fit(X, y)
        return self

    def transform(self, X):
        if self.estimator:
            X_r = self.estimator.transform(X)
            return X_r
        else:
            return X
        
class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator = RandomForestClassifier()):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

def toeic2cefr(row):
    if row.L2_proficiency < 120:
        return 'Low'
    elif row.L2_proficiency < 255:
        return 'A1'
    elif row.L2_proficiency < 550:
        return 'A2'
    elif row.L2_proficiency < 785:
        return 'B1'
    else:
        return 'B2+'

def display_scores(scores):
    print('All scores:\n', np.sort(scores), '\n')
    print('Mean:', scores.mean())
    print('Median:', np.median(scores))
    print('SD:', scores.std())

def write_pipeline_scores(scores, filename):
    file = open(filename, 'a')
    scores_str = [str(score) for score in scores]
    file.write('All scores:\n' + str(', '.join(scores_str)) + '\n')
    file.write('Mean:' + str(scores.mean()) + '\n')
    file.write('Median:' + str(np.median(scores)) + '\n')
    file.write('SD:' + str(scores.std()) + '\n')
    file.close()

def write_best_grid_search_score(grid_search_results, filename):
    file = open(filename, 'a')
    score = np.sqrt(abs(grid_search_results.best_score_))
    file.write('RMSE:\n' + str(score) + '\n')
    file.close()