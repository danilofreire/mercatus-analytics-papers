library(reticulate)
library(tidyverse)

# This code borrows from Muchlinski et al (2016)
# See: https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/KRKWK8/S8SE0K&version=1.0

# Set working directory
setwd("~/Documents/github/mercatus-analytics-papers")

# Load data and select variables
data <- read_csv("civil-war-data.csv") %>%
  select(warstds, ager, agexp, anoc, army85, autch98, auto4,
         autonomy, avgnabo, centpol3, coldwar, decade1, decade2,
         decade3, decade4, dem, dem4, demch98, dlang, drel,
         durable, ef, ef2, ehet, elfo, elfo2, etdo4590,
         expgdp, exrec, fedpol3, fuelexp, gdpgrowth, geo1, geo2,
         geo34, geo57, geo69, geo8, illiteracy, incumb, infant,
         inst, inst3, life, lmtnest, ln_gdpen, lpopns, major,
         manuexp, milper, mirps0, mirps1, mirps2, mirps3, nat_war,
         ncontig, nmgdp, nmdp4_alt, numlang, nwstate, oil, p4mchg,
         parcomp, parreg, part, partfree, plural, plurrel, pol4,
         pol4m, pol4sq, polch98, polcomp, popdense, presi, pri,
         proxregc, ptime, reg, regd4_alt, relfrac, seceduc,
         second, semipol3, sip2, sxpnew, sxpsq, tnatwar, trade,
         warhist, xconst)

# Convert dependent variable into factor
# data$warstds <- as.factor(data$warstds)

# Fearon and Laitin (2003)

# Select variables
fl_data <- data %>%
  select(warstds, warhist, ln_gdpen, lpopns,
         lmtnest, ncontig, oil, nwstate, inst3,
         pol4, ef, relfrac)

# Independent variables, dependent variable
fl_x <- fl_data %>% select(-warstds)
fl_y <- fl_data %>% select(warstds)


### Python
repl_python()
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from collections import Counter

X_train, X_test, y_train, y_test = train_test_split(r.fl_x, r.fl_y, train_size=0.75, test_size=0.25, stratify=r.fl_y) 
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

oversample = SMOTETomek(random_state=48924)
X_train_SMOTE, y_train_SMOTE = oversample.fit_resample(X_train, y_train)
Counter(y_train_SMOTE)

# TPOT
from tpot import TPOTClassifier
tpot = TPOTClassifier(max_time_mins=10, cv=5, random_state=48924, scoring='roc_auc', verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_predict),
                        columns=['Predicted NO', 'Predicted YES'],
                        index=['Actual NO', 'Actual YES']) 
print(conf_mat) 


# H2O
import h2o
from h2o.automl import H2OAutoML
h2o.init()

r.fl_data_h2o = h2o.H2OFrame(r.fl_data) 
r.fl_data_h2o["warstds"] = r.fl_data_h2o["warstds"].asfactor()

# set the predictor names and the response column name
predictors = ["warhist", "ln_gdpen", "lpopns", "lmtnest",
              "ncontig", "oil", "nwstate", "inst3",
              "pol4", "ef", "relfrac"]
response = "warstds"

# split into train and validation sets
train, test = r.fl_data_h2o.split_frame(ratios = [.75], seed = 48924)
x = train.columns
y = "warstds"
x.remove(y)

# run the model
aml = H2OAutoML(max_runtime_secs=600, sort_metric="AUC", seed=48924)
aml.train(x=x, y=y, training_frame=train)
perf = aml.leader.model_performance(valid)
round(perf.auc(), 4)
aml.leader.confusion_matrix()
preds = aml.predict(valid)
lb = h2o.automl.get_leaderboard(aml)
lb.head(2)

# Autogluon
import autogluon as ag



exit
