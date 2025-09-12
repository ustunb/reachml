import numpy as np
import xgboost

import reachml

# load dataset and train xgboost classifier
X, y = reachml.datasets.givemecredit_cts_slim(label=[0, 1])
model = xgboost.XGBClassifier().fit(X, y)

# create simple action set
A = reachml.ActionSet(X)

# Immutable features
A["HistoryOfDelinquency"].actionable = False
A["HistoryOfLatePayment"].actionable = False

# MonthlyIncome can change +- 5000
A["MonthlyIncome"].step_ub = 5000
A["MonthlyIncome"].step_lb = -5000

# CreditLineUtilization can change +- 1
A["CreditLineUtilization"].step_ub = -1
A["CreditLineUtilization"].step_lb = 1

# Min CreditLineUtilization is 0
A["CreditLineUtilization"].lb = 0

# Calculate responsiveness scores
# Since some features are continuous, we set sample size: n = 100
scorer = reachml.ResponsivenessScorer(A)
scores = scorer(X, model, n=100)

# List of adverse outcome indices
rejected = np.where(model.predict(X) == 0)[0]

# Plot output (pass in index)
scorer.plot(x_idx=rejected[0])
