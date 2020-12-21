# Required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score

# Load data
fl_x = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/fl_x.csv")
fl_y = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/fl_y.csv")
fl_data = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/fl_data.csv")

X_train, X_test, y_train, y_test = train_test_split(fl_x, fl_y, train_size=0.75, test_size=0.25, stratify=fl_y, random_state = 48924)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

## H2O
import h2o
from h2o.automl import H2OAutoML
h2o.init()

fl_data_h2o = h2o.H2OFrame(fl_data)
fl_data_h2o["onset"] = fl_data_h2o["onset"].asfactor()

# Set the predictor names and the response column name
predictors = ["warl", "gdpenl", "lpopl1", "lmtnest",
              "ncontig", "oil", "nwstate", "instab",
              "polity2l", "ethfrac", "relfrac"]
response = "onset"

# Split into training and test datasets
train, test = fl_data_h2o.split_frame(ratios = [.75], seed = 48924)
x = train.columns
y = "onset"
x.remove(y)

# Run the model
aml = H2OAutoML(max_runtime_secs=3600, sort_metric="AUC", seed=48924)
aml.train(x=x, y=y, training_frame=train)
aml.leader.confusion_matrix()
perf = aml.leader.model_performance(test)
round(perf.auc(), 3)

## mljar-supervised
from supervised.automl import AutoML
automl = AutoML(mode="Compete", golden_features=False, total_time_limit=3600, random_state=48924)
automl.fit(X_train, y_train)

y_pred_prob = automl.predict_proba(X_test)
roc_auc_score(y_test, y_pred_prob)

predictions = automl.predict(X_test)
print(predictions.head())

## TPOT
from tpot import TPOTClassifier
tpot = TPOTClassifier(max_time_mins=60, cv=5, random_state=48924, scoring='roc_auc', verbosity=2)
tpot.fit(X_train, y_train)
print(round(tpot.score(X_test, y_test), 3))
