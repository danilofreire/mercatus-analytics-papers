# Required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score

## Fearon and Laitin (2003)

# Load data
fl_x = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/fl_x.csv")
fl_y = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/fl_y.csv")
fl_data = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/fl_data.csv")

X_train, X_test, y_train, y_test = train_test_split(fl_x, fl_y, train_size=0.75, test_size=0.25, stratify=fl_y, random_state = 8305)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

## AutoKeras
import autokeras as ak
from tensorflow.keras.metrics import AUC
import kerastuner

search = ak.StructuredDataClassifier(seed=8305,max_trials=10, overwrite=True, metrics=["accuracy","AUC"], objective=kerastuner.Objective("val_auc", direction="max")) 
search.fit(x=X_train, y=y_train, verbose=1)
model = search.export_model()
model.summary()
loss, acc, auc = search.evaluate(X_test, y_test, verbose=0)
print(round(auc, 3))

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
train, test = fl_data_h2o.split_frame(ratios = [.75], seed = 8305)
x = train.columns
y = "onset"
x.remove(y)

# Run the model
aml = H2OAutoML(max_runtime_secs=600, sort_metric="AUC", seed=8305)
aml.train(x=x, y=y, training_frame=train)
aml.leader.confusion_matrix()
perf = aml.leader.model_performance(test)
round(perf.auc(), 3)

## mljar-supervised
from supervised.automl import AutoML
automl = AutoML(total_time_limit=600, random_state=8305)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
roc_auc_score(y_test, predictions)
y_pred_prob = automl.predict_proba(X_test)

## TPOT
from tpot import TPOTClassifier
tpot = TPOTClassifier(max_time_mins=10, cv=5, random_state=8305, scoring='roc_auc', verbosity=2)
tpot.fit(X_train, y_train)
print(round(tpot.score(X_test, y_test), 3))


## Collier and Hoeffler (2004)

# Load data
ch_data = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/ch_data.csv")
ch_x = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/ch_x.csv")
ch_y = pd.read_csv("~/Documents/github/mercatus-analytics-papers/automl-paper/ch_y.csv")

X_train, X_test, y_train, y_test = train_test_split(ch_x, ch_y, train_size=0.75, test_size=0.25, stratify=ch_y, random_state = 8305)
y_test = np.ravel(y_test)
y_train = np.ravel(y_train)

## AutoKeras
import autokeras as ak
from tensorflow.keras.metrics import AUC
import kerastuner

search = ak.StructuredDataClassifier(seed=8305,max_trials=20, overwrite=True, metrics=["accuracy","AUC"], objective=kerastuner.Objective("val_auc", direction="max")) 
search.fit(x=X_train, y=y_train, verbose=1)
model = search.export_model()
model.summary()
loss, acc, auc = search.evaluate(X_test, y_test, verbose=0)
print(round(auc, 3))

## H2O
import h2o
from h2o.automl import H2OAutoML
h2o.init()

ch_data_h2o = h2o.H2OFrame(ch_data)
ch_data_h2o["warsa"] = ch_data_h2o["warsa"].asfactor()

# Set the predictor names and the response column name
predictors = ["sxp", "sxp2", "secm", "gy1", "peace",
              "geogia", "lnpop", "frac", "etdo4590"]
response = "warsa"

# Split into training and test datasets
train, test = ch_data_h2o.split_frame(ratios = [.75], seed = 8305)
x = train.columns
y = "warsa"
x.remove(y)

# Run the model
aml = H2OAutoML(max_runtime_secs=600, sort_metric="AUC", seed=8305)
aml.train(x=x, y=y, training_frame=train)
aml.leader.confusion_matrix()
perf = aml.leader.model_performance(test)
round(perf.auc(), 3)

## mljar-supervised
from supervised.automl import AutoML
automl = AutoML(mode="Compete", total_time_limit=600, random_state=8305)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
roc_auc_score(y_test, predictions)
y_pred_prob = automl.predict_proba(X_test)

## TPOT
from tpot import TPOTClassifier
tpot = TPOTClassifier(max_time_mins=10, cv=5, random_state=8305, scoring='roc_auc', verbosity=2)
tpot.fit(X_train, y_train)
print(round(tpot.score(X_test, y_test), 3))

