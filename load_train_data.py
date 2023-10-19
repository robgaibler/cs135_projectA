
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
x_test_NF = vectorizer.transform(te_text_list).toarray()
y_train_N = y_train_data['is_positive_sentiment'].values

N, F = x_train_NF.shape

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# define models and parameters
model = LogisticRegression()
solvers = ['lbfgs']
penalty = ['l2']
c_values = np.logspace(-9, 6, 30)
# define grid search and base the success metric on AUROC
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)
grid_result = grid_search.fit(x_train_NF, y_train_N)

# summarize results with the best hyperparamters 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

C=grid_result.best_params_['C']
pen=grid_result.best_params_['penalty']
sol=grid_result.best_params_['solver']

bestmodel= LogisticRegression(C=C, solver=sol, penalty=pen)
bestmodel.fit(x_train_NF, y_train_N)
prediction = bestmodel.predict(x_test_NF)
predictionprob=bestmodel.predict_proba(x_test_NF)[:,1]

yproba1_test = np.array(predictionprob)
np.savetxt("yproba1_test.txt", yproba1_test)
