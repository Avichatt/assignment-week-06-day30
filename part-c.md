Part C 

Q1.

Logistic Regression is an algorithmic approach used in statistics and machine learning to predict the probability of a categorical dependent variable. 

Despite having the word "regression" in its name, **it is a classification algorithm**. It is typically used for binary classification (predicting whether something belongs to class 0 or class 1, Yes or No, True or False). It uses a logistic function (sigmoid curve) to model a binary dependent variable, predicting a probability value mapped between 0 and 1. 

----------------------------------------------------------------------------------------------

Q2.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume X and y have already been defined/extracted from a DataFrame
# X = features (e.g., Age, EstimatedSalary)
# y = target (e.g., Purchased)

# 1. Perform train-test split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 2. Perform feature scaling
scaler = StandardScaler()

# Fit the scaler on the train set and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set using the already fitted scaler to prevent data leakage
X_test_scaled = scaler.transform(X_test)


-----------------------------------------------------------------------------------------------

Q3.


Confusion Matrix is a specific table structure used to evaluate the performance of a classification model. It represents a summary of prediction results on a classification problem. 

For a binary classification problem, it is a 2x2 matrix representing four important metrics:
1. True Positives (TP): The model correctly predicted the positive class (e.g., correctly predicted they will purchase).
2. True Negatives (TN): The model correctly predicted the negative class (e.g., correctly predicted they will not purchase).
3. False Positives (FP) (Type I Error): The model incorrectly predicted the positive class (e.g., predicted they will purchase, but they didn't).
4. False Negatives (FN) (Type II Error): The model incorrectly predicted the negative class (e.g., predicted they won't purchase, but they actually did). 

It is highly useful for measuring metrics that go beyond simple accuracy, like Precision, Recall, and F1-score.
