# Install and load required packages
if (!require("haven")) {
    install.packages("haven")
}
if (!require("tidyverse")) {
    install.packages("tidyverse")
}

### Data wrangling

# Set working directory
setwd("~/Documents/github/mercatus-analytics-papers/automl-paper")

# Fearon and Laitin (2003)

# Load data and select variables
load("fl.three.RData")

fl_data <- fl.three %>%
  select(onset, warl, gdpenl, lpopl1,
         lmtnest, ncontig, Oil, nwstate, instab,
         polity2l, ethfrac, relfrac) %>%
  mutate(onset = if_else(onset >= 1, 1, onset),
         onset = as.factor(onset),
         oil = Oil) %>%
  select(-Oil)

# Independent variables, dependent variable
fl_x <- fl_data %>% select(-onset)
fl_y <- fl_data %>% select(onset)

# Train, test data
fl_train <- fl_data %>% sample_frac(.75)
fl_test <- anti_join(fl_data, fl_train)

# Write csv
write_csv(fl_x, "fl_x.csv")
write_csv(fl_y, "fl_y.csv")
write_csv(fl_train, "fl_train.csv")
write_csv(fl_test, "fl_test.csv")
write_csv(fl_data, "fl_data.csv")

# Collier and Hoeffler (2004)

# Load data
ch <- haven::read_dta("~/Documents/github/mercatus-analytics-papers/automl-paper/G&G.dta")

# Select variables
ch_data <- ch %>%
  select(warsa, sxp, sxp2, secm, gy1, peace,  geogia, lnpop, frac, etdo4590) %>%
  drop_na()

# Independent variables, dependent variable
ch_x <- ch_data %>% select(-warsa)
ch_y <- ch_data %>% select(warsa)

# Train, test data
ch_train <- ch_data %>% sample_frac(.75)
ch_test <- anti_join(ch_data, ch_train)

# Write csv
write_csv(ch_x, "ch_x.csv")
write_csv(ch_y, "ch_y.csv")
write_csv(ch_train, "ch_train.csv")
write_csv(ch_test, "ch_test.csv")
write_csv(ch_data, "ch_data.csv")
