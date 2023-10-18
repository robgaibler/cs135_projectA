
import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data_dir = 'data_reviews'
    x_train_data = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_data = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_data = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    tr_text_list = x_train_data['text'].values.tolist()
    te_text_list = x_test_data['text'].values.tolist()

    # Part 1A: Vectorize and clean data

    vectorizer = CountVectorizer(stop_words='english', token_pattern="[a-z]+", 
                                 binary=True)
    
    x_train_NF = vectorizer.fit_transform(tr_text_list).toarray()
    x_test_NF = vectorizer.fit_transform(te_text_list).toarray()
    y_train_N = y_train_data['is_positive_sentiment'].values

    N, F = x_train_NF.shape

    # Part 1B:

    num_splits = 5

    num_c_vals = 20
    C_vals = np.logspace(0, 2, num_c_vals)

    error_grid = np.zeros((num_c_vals, num_splits))

    kf = KFold(n_splits=num_splits)

    for i in range(C_vals.size):
        for j, (train_index, test_index) in enumerate(kf.split(x_train_NF)):
            model = LogisticRegression(C=C_vals[i], solver='lbfgs', max_iter=500)

            model.fit(x_train_NF[train_index], y_train_N[train_index])

            prediction = model.predict(x_train_NF[test_index])

            error_grid[i][j] = mean_absolute_error(y_train_N[test_index], prediction)