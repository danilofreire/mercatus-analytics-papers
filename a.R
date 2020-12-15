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
data$warstds <- as.factor(data$warstds)

# Fearon and Laitin (2003)

# Select variables
fl_data <- data %>%
  select(warstds, warhist, ln_gdpen, lpopns,
         lmtnest, ncontig, oil, nwstate, inst3,
         pol4, ef, relfrac)

set.seed(48924)
fl_train <- fl_data %>% sample_frac(.7)
fl_test <- anti_join(fl_data, fl_train)
fl_xtrain <- fl_train %>% select(-warstds)
fl_ytrain <- fl_train %>% select(warstds)
fl_xtest <- fl_test %>% select(-warstds)
fl_ytest <- fl_test %>% select(warstds)

fl_x <- fl_data %>% select(-warstds)
fl_y <- fl_data %>% select(warstds)

np <- import("numpy", convert = FALSE)
fl_ytrain_py <- np$ravel(fl_ytrain)

library("RemixAutoML")

# m1 <- AutoXGBoostClassifier()


repl_python()

from tpot import TPOTClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from collections import Counter

X_train, X_test, y_train, y_test = train_test_split(r.fl_x, r.fl_y, train_size=0.75, test_size=0.25, stratify=r.fl_y) 
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

oversample = SMOTE(random_state=48924)
X_train_SMOTE, y_train_SMOTE = oversample.fit_resample(X_train, y_train)


tpot = TPOTClassifier(generations=4, population_size=100, cv=4, random_state=48924, verbosity=2)
tpot.fit(X_train_SMOTE, y_train_SMOTE)
print(tpot.score(X_test, y_test))

y_predict = tpot.predict(X_test)
y_prob = [probs[1] for probs in tpot.predict_proba(X_test)]

print("Test accuracy: %s\n"%(accuracy_score(y_test, y_predict).round(2)))

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_predict),
                        columns=['Predicted NO', 'Predicted YES'],
                        index=['Actual NO', 'Actual YES']) 
print(conf_mat) 

# Compute area under the curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


exit
