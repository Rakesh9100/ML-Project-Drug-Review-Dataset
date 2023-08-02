# Import pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Import warnings to control warning messages
import warnings

# Import BeautifulSoup for web scraping and parsing HTML/XML
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Import SimpleImputer for handling missing values
from sklearn.impute import SimpleImputer

# Import ConvergenceWarning for optimization convergence issues
from sklearn.exceptions import ConvergenceWarning

# Import TfidfVectorizer for text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Import LabelEncoder for categorical label encoding
from sklearn.preprocessing import LabelEncoder

# Import LinearRegression and LogisticRegression for regression and classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Import catboost for gradient boosting
import catboost as cbt

# Import optuna for hyperparameter optimization
import optuna

# Import MLPClassifier for training MLP neural networks
from sklearn.neural_network import MLPClassifier

# Import DecisionTreeClassifier for training decision tree models
from sklearn.tree import DecisionTreeClassifier

# Import gensim for topic modeling and document similarity
import gensim

# Import evaluation metrics and visualization functions
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
)

# Import RandomForestRegressor for regression with random forests
from sklearn.ensemble import RandomForestRegressor

# Import RandomizedSearchCV for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# Import seaborn and matplotlib.pyplot for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Import PorterStemmer for word stemming
from gensim.parsing.porter import PorterStemmer

# Import StandardScaler and MinMaxScaler for feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import RandomOverSampler for oversampling imbalanced datasets
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Import Sequential, Dense, LSTM, and Embedding for building neural network models
from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    LSTM,
    GRU,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
)

# Define the data types for each column
dtypes = {
    "Unnamed: 0": "int32",
    "drugName": "category",
    "condition": "category",
    "review": "category",
    "rating": "float16",
    "date": "string",
    "usefulCount": "int16",
}

# Read the training dataset from a TSV file, specifying the column separator as tab
train_df = pd.read_csv(
    r"datasets\drugsComTrain_raw.tsv", quoting=2, dtype=dtypes, on_bad_lines="skip"
)


# Sample a fraction (80%) of the training dataset with a random seed of 42
train_df = train_df.sample(frac=0.8, random_state=42)

# Read the testing dataset from a TSV file, specifying the column separator as tab
test_df = pd.read_csv(
    r"datasets\drugsComTest_raw.tsv", quoting=2, dtype=dtypes, on_bad_lines="skip"
)

# Convert the "date" column to datetime format in the training and testing datasets
train_df["date"], test_df["date"] = pd.to_datetime(
    train_df["date"], format="%d-%b-%y"
), pd.to_datetime(test_df["date"], format="%d-%b-%y")

## Extracting day, month, and year into separate columns
for df in [train_df, test_df]:
    df["day"] = df["date"].dt.day.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")
    df["year"] = df["date"].dt.year.astype("int16")

## Suppressing MarkupResemblesLocatorWarning, FutureWarning and ConvergenceWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


## Defining function to decode HTML-encoded characters
def decode_html(text):
    decoded_text = BeautifulSoup(text, "html.parser").get_text()
    return decoded_text


## Applying the function to the review column
train_df["review"], test_df["review"] = train_df["review"].apply(decode_html), test_df[
    "review"
].apply(decode_html)

## Dropped the original date column and removed the useless column
train_df, test_df = [
    df.drop("date", axis=1).drop(df.columns[0], axis=1) for df in (train_df, test_df)
]

## Handling the missing values and assigning old column names
train_imp, test_imp = [
    pd.DataFrame(
        SimpleImputer(strategy="most_frequent").fit_transform(df), columns=df.columns
    )
    for df in (train_df, test_df)
]

# Assign new column names to the training dataset
train_imp.columns = [
    "drugName",
    "condition",
    "review",
    "rating",
    "usefulCount",
    "day",
    "month",
    "year",
]

# Assign new column names to the testing dataset
test_imp.columns = [
    "drugName",
    "condition",
    "review",
    "rating",
    "usefulCount",
    "day",
    "month",
    "year",
]

# Tokenize the text in the "review" column of the training dataset
train_imp["tokenized_text"] = [
    gensim.utils.simple_preprocess(line, deacc=True) for line in train_imp["review"]
]

# Tokenize the text in the "review" column of the testing dataset
test_imp["tokenized_text"] = [
    gensim.utils.simple_preprocess(line, deacc=True) for line in test_imp["review"]
]

# Create an instance of the PorterStemmer
porter_stemmer = PorterStemmer()

# Perform stemming on the tokenized text in the training dataset
train_imp["stemmed_tokens"] = [
    [porter_stemmer.stem(word) for word in tokens]
    for tokens in train_imp["tokenized_text"]
]

# Perform stemming on the tokenized text in the testing dataset
test_imp["stemmed_tokens"] = [
    [porter_stemmer.stem(word) for word in tokens]
    for tokens in test_imp["tokenized_text"]
]

##Applying Word2vec
from gensim.models import Word2Vec

# Skip-gram model (sg = 1)
size = 1000
window = 3
min_count = 1  # The minimum count of words to consider when training the model; words with occurrence less than this count will be ignored.
workers = 3
sg = 1  # The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.

## Merging values from both train and test dataframe
stemmed_tokens_train = pd.Series(train_imp["stemmed_tokens"]).values
stemmed_tokens_test = pd.Series(test_imp["stemmed_tokens"]).values
stemmed_tokens_merged = np.append(stemmed_tokens_train, stemmed_tokens_test, axis=0)
w2vmodel = Word2Vec(
    stemmed_tokens_merged,
    min_count=min_count,
    vector_size=size,
    workers=workers,
    window=window,
    sg=sg,
)

### Store the vectors for train data in following file
index = 0
word2vec_filename = "train_review_word2vec.csv"
with open(word2vec_filename, "w") as word2vec_file:
    for i in range(129038):
        model_vector = (
            np.mean(
                [w2vmodel.wv[token] for token in train_imp["stemmed_tokens"][i]], axis=0
            )
        ).tolist()
        if index == 0:
            header = ",".join(str(ele) for ele in range(1000))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        index += 1
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:
            line1 = ",".join([str(vector_element) for vector_element in model_vector])
        word2vec_file.write(line1)
        word2vec_file.write("\n")

review_vector = pd.read_csv(r"train_review_word2vec.csv")

### Store the vectors for test data in following file

index = 0
word2vec_filename = "test_review_word2vec.csv"
with open(word2vec_filename, "w") as word2vec_file:
    for i in range(53766):
        model_vector = (
            np.mean(
                [w2vmodel.wv[token] for token in test_imp["stemmed_tokens"][i]], axis=0
            )
        ).tolist()
        if index == 0:
            header = ",".join(str(ele) for ele in range(1000))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        index += 1
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:
            line1 = ",".join([str(vector_element) for vector_element in model_vector])
        word2vec_file.write(line1)
        word2vec_file.write("\n")
reivew_vector1 = pd.read_csv(r"test_review_word2vec.csv")

## Joining vector and dropping necessary columns
train_imp = pd.concat([train_imp, review_vector], axis="columns")
train_imp.drop(
    ["review", "tokenized_text", "stemmed_tokens"], axis="columns", inplace=True
)
test_imp = pd.concat([test_imp, reivew_vector1], axis="columns")
test_imp.drop(
    ["review", "tokenized_text", "stemmed_tokens"], axis="columns", inplace=True
)

## Encoding the categorical columns
for i in ["drugName", "condition"]:
    train_imp[i] = LabelEncoder().fit_transform(train_imp[i])
    test_imp[i] = LabelEncoder().fit_transform(test_imp[i])

## Converting the data types of columns to reduce the memory usage
train_imp, test_imp = train_imp.astype("float16"), test_imp.astype("float16")
train_imp[["drugName", "condition", "usefulCount", "year"]] = train_imp[
    ["drugName", "condition", "usefulCount", "year"]
].astype("int16")
test_imp[["drugName", "condition", "usefulCount", "year"]] = test_imp[
    ["drugName", "condition", "usefulCount", "year"]
].astype("int16")
train_imp[["rating"]] = train_imp[["rating"]].astype("float16")
test_imp[["rating"]] = test_imp[["rating"]].astype("float16")
train_imp[["day", "month"]] = train_imp[["day", "month"]].astype("int8")
test_imp[["day", "month"]] = test_imp[["day", "month"]].astype("int8")

"""
Implementation of pycaret on the Preprocessed data (given datasets)

requirements: pip install pycaret

Regression

PyCaret’s Regression Module is a supervised machine learning module that is used for estimating the relationships between a dependent variable (often called the ‘outcome variable’, or ‘target’) and one or more independent variables (often called ‘features’, ‘predictors’, or ‘covariates’). 
The objective of regression is to predict continuous values such as predicting sales amount, predicting quantity, predicting temperature, etc. 

From line 297 to 325 depict the Implementation of Accuracy enhancement using multiple regression models at a time 
"""
# setup
from pycaret.regression import *

s = setup(train_imp, target="rating")
# s = setup(test_imp, target = 'rating')

# compare models
best = compare_models()

print(best)

# analyze models
evaluate_model(best)

plot_model(best, plot="residuals")

plot_model(best, plot="feature")

# predictions
predict_model(best)

predictions = predict_model(best, data=train_df)
predictions.head()

# save the model
save_model(best, "my_best_pipeline")

# load the saved model
loaded_model = load_model("my_best_pipeline")
print(loaded_model)


# print(train_imp.iloc[:,:15].dtypes)
# print(test_imp.iloc[:,:15].dtypes)

# Split the training dataset into feature variables (X_train) and the target variable (Y_train)
X_train, Y_train = train_imp.drop("rating", axis=1), train_imp["rating"]

# Split the testing dataset into feature variables (X_test) and the target variable (Y_test)
X_test, Y_test = test_imp.drop("rating", axis=1), test_imp["rating"]

# Convert the column names to string type in the training dataset
X_train.columns = X_train.columns.astype(str)

# Convert the column names to string type in the testing dataset
X_test.columns = X_test.columns.astype(str)

# Plot a scatter chart between Drug Name and Ratings in the testing dataset
# Encode the drug names using LabelEncoder for visualization purposes
plt.scatter(LabelEncoder().fit_transform(test_df.drugName), test_df.rating)
plt.xlabel("Drug Name")
plt.ylabel("Ratings")
plt.title("Scatter Plot: Drug Name vs Ratings (Testing Data)")
plt.show()

# Generate multiple scatter plots and histograms for the training dataset
feature = ["drugName", "condition", "rating", "usefulCount"]
pd.plotting.scatter_matrix(train_imp[feature])
plt.suptitle("Scatter Matrix For Training DataSet")
plt.show()

plt.title("Drug Name Histogram (Training Dataset)")
plt.hist(train_imp["drugName"], bins=50)
plt.xlabel("Bins")
plt.ylabel("Drug Name")
plt.show()


##### LinearRegression regression algorithm #####
##### EDA

##### 1) Summary and Stats

# a) Checking Null Values

print("Null Values in Train Data:\n", X_train.isnull().sum())
print("Null Values in Test Data:\n", X_test.isnull().sum())

# b) Checking the shape of the data

print("Shape of Train Data:", X_train.shape)
print("Shape of Test Data:", X_test.shape)

# c) Zero Counts

print("Zero Counts in Train Data:\n", (X_train == 0).sum())
print("Zero Counts in Test Data:\n", (X_test == 0).sum())

##### 2) Visualizations

# a) Box Plot

plt.figure(figsize=(10, 6))
sns.boxplot(x="rating", data=train_imp)
plt.title("Box Plot of Rating")
plt.show()

# b) Class Imbalance

plt.figure(figsize=(10, 6))
sns.countplot(x="rating", data=train_imp)
plt.title("Class Imbalance of Rating")
plt.show()

### Over Sampling to handle Class Imbalance

ros = RandomOverSampler(random_state=0)
X_train, Y_train = ros.fit_resample(X_train, Y_train)
plt.hist(Y_train, bins=10)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(x="rating", data=train_imp)
plt.title("Class Imbalance of Rating after OverSampling")
plt.show()

##################################################

# Create an instance of the LinearRegression algorithm
linear = LinearRegression()

# Fit the LinearRegression model on the training data
linear.fit(X_train, Y_train)

# Make predictions on the training and testing data using the LinearRegression model
line_train = linear.predict(X_train)
line_test = linear.predict(X_test)

# Print the evaluation metrics for Linear Regression
print("Linear Regression Metrics:")
print("MSE for training: ", mean_squared_error(Y_train, line_train))
print("MSE for testing: ", mean_squared_error(Y_test, line_test))
print("R2 score for training: ", r2_score(Y_train, line_train))
print("R2 score for testing: ", r2_score(Y_test, line_test))


# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, line_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Linear Regression - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, line_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Linear Regression - Testing Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs true values for both training and testing sets
plt.figure(figsize=(8, 6))
plt.scatter(Y_train, line_train, alpha=0.3, label="Training")
plt.scatter(Y_test, line_test, alpha=0.3, label="Testing")
plt.plot([0, 10], [0, 10], linestyle="--", color="k", label="Perfect prediction")
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Linear regression - Training and Testing Sets Scatter Plot")
plt.legend()
plt.show()

# Plotting the residual plot for testing data
plt.scatter(line_test, line_test - Y_test, c="g", s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=10)
plt.xlabel("Predicted Ratings")
plt.ylabel("Residuals")
plt.title("Linear Regression - Testing Data Residual Plot")
plt.show()

for i in range(len(line_test)):
    line_test[i] = line_test[i].round()

# Pie Chart
correct_predictions = sum(Y_test == line_test)
incorrect_predictions = len(Y_test) - correct_predictions

labels = ["Correct Predictions", "Incorrect Predictions"]
sizes = [correct_predictions, incorrect_predictions]
colors = ["green", "red"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)

plt.axis("equal")  # Equal aspect ratio ensures a circular pie chart
plt.title("Prediction Accuracy")

# Bar Chart
prediction_column = "predicted_rating"
target_column = "actual_rating"

# Get the predicted ratings and actual ratings from the dataset
predicted_ratings = line_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Count the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[actual_ratings == rounded_predictions], bins=bins
)

# Create the bar chart
plt.bar(range(1, 11), correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Bar Chart of Correct Predictions")

# Display the bar chart
plt.show()

# Histogram
plt.hist(Y_test, bins="auto", alpha=0.5, color="blue", label="Actual")
plt.hist(line_test, bins="auto", alpha=0.5, color="red", label="Predicted")

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Histogram")
plt.legend()
plt.grid(True)
plt.show()

# Line Chart
predicted_ratings = line_test
actual_ratings = Y_test


# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create a line chart for the number of correct predictions
x = range(1, 11)

plt.plot(x, correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Line Chart of Correct Predictions")
plt.show()

# Area Chart
predicted_ratings = line_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create an area chart for the number of correct predictions
x = range(1, 11)

plt.fill_between(x, correct_predictions, color="blue", alpha=0.3)
plt.plot(x, correct_predictions, color="blue", linewidth=2)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Area Chart of Correct Predictions")

plt.show()

##### ANN algorithm #####
from sklearn.neural_network import MLPClassifier

# Create an instance of the MLPClassifier model
model = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)

# Fit the MLPClassifier model on the training data
model.fit(X_train, Y_train)

# Make predictions on the training and testing data using the MLPClassifier model
model_train = model.predict(X_train)
model_test = model.predict(X_test)

# Calculate accuracy scores for the training and testing predictions
train_accuracy = accuracy_score(model_train, Y_train)
test_accuracy = accuracy_score(model_test, Y_test)

# Print the evaluation metrics for the MLPClassifier model
print("\nANN METRICS:")
print("Accuracy for training: ", train_accuracy)
print("Accuracy for testing: ", test_accuracy)
print("MSE for training: ", mean_squared_error(Y_train, model_train))
print("MSE for testing: ", mean_squared_error(Y_test, model_test))
print("R2 score for training: ", r2_score(Y_train, model_train))
print("R2 score for testing: ", r2_score(Y_test, model_test))


# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, model_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("ANN - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, model_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("ANN - Testing Data Scatter Plot")
plt.show()

# Plotting the Accuracy Plot
plt.plot(["Training", "Testing"], [train_accuracy, test_accuracy], marker="o")
plt.title("ANN Accuracy")
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.show()

# Plotting the confusion matrix
cm = confusion_matrix(Y_test, model_test, labels=model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()
plt.title("ANN Confusion Matrix")
plt.show()

##### ADABOOST algorithm #####
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

# Generate a synthetic dataset for training
X_train, Y_train = make_classification(
    n_samples=1006,
    n_features=1006,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

# Create an instance of the AdaBoostClassifier model
clf = AdaBoostClassifier(n_estimators=100, random_state=0)

# Fit the AdaBoostClassifier model on the training data
clf.fit(X_train, Y_train)

# Make predictions on the training and testing data using the AdaBoostClassifier model
ada_train = clf.predict(X_train)
ada_test = clf.predict(X_test)

# Calculate accuracy scores for the training and testing predictions
ada_train_accuracy = accuracy_score(ada_train, Y_train)
ada_test_accuracy = accuracy_score(ada_test, Y_test)

# Print the evaluation metrics for the AdaBoostClassifier model
print("\nAdaBOOST METRICS:")
print("Accuracy for training: ", ada_train_accuracy)
print("Accuracy for testing: ", ada_test_accuracy)
print("MSE for training: ", mean_squared_error(Y_train, ada_train))
print("MSE for testing: ", mean_squared_error(Y_test, ada_test))
print("R2 score for training: ", r2_score(Y_train, ada_train))
print("R2 score for testing: ", r2_score(Y_test, ada_test))


# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, ada_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("AdaBoost - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, ada_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("AdaBoost - Testing Data Scatter Plot")
plt.show()

# Plotting the Accuracy Plot
plt.plot(["Training", "Testing"], [ada_train_accuracy, ada_test_accuracy], marker="o")
plt.title("AdaBoost Accuracy")
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.show()

# Plotting the confusion matrix
cm = confusion_matrix(Y_test, ada_test, labels=clf.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_).plot()
plt.title("AdaBoost Confusion Matrix")
plt.show()

##### XGBOOST ####
import xgboost

# Create an instance of the XGBRegressor model with specified parameters
model = xgboost.XGBRegressor(
    n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8
)

# Fit the XGBRegressor model on the training data
model.fit(X_train, Y_train)

# Make predictions on the training and testing data using the XGBRegressor model
xg_train = model.predict(X_train)
xg_test = model.predict(X_test)

# Print the evaluation metrics for XGBoost
print("XGBoost Metrics:")
print("MSE for training: ", mean_squared_error(Y_train, xg_train))
print("MSE for testing: ", mean_squared_error(Y_test, xg_test))
print("R2 score for training: ", r2_score(Y_train, xg_train))
print("R2 score for testing: ", r2_score(Y_test, xg_test))


# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, xg_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("XGBoost Regression - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, xg_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("XGBoost Regression - Testing Data Scatter Plot")
plt.show()

##### LGBM ####


from lightgbm import LGBMRegressor

# Create an instance of the LGBMRegressor model
lgbmodel = LGBMRegressor()

# Fit the LGBMRegressor model on the training data
lgbmodel.fit(X_train, Y_train)

# Make predictions on the training and testing data using the LGBMRegressor model
lgb_train = lgbmodel.predict(X_train)
lgb_test = lgbmodel.predict(X_test)

# Print the evaluation metrics for LGBMRegressor
print("LGBM Metrics:")
print("MSE for training: ", mean_squared_error(Y_train, lgb_train))
print("MSE for testing: ", mean_squared_error(Y_test, lgb_test))
print("R2 score for training: ", r2_score(Y_train, lgb_train))
print("R2 score for testing: ", r2_score(Y_test, lgb_test))


# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, lgb_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("LGBM Regression - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, lgb_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("LGBM Regression - Testing Data Scatter Plot")
plt.show()


##### SVR #####

from sklearn import svm

# Create an instance of the Support Vector Machine (SVM) model for regression
svm_model = svm.SVR()

# Fit the SVM model on the training data
svm_model.fit(X_train, Y_train)

# Make predictions on the training and testing data using the SVM model
svm_train = svm_model.predict(X_train)
svm_test = svm_model.predict(X_test)

# Print the evaluation metrics for SVM regression
print("SVM Regression Metrics:")
print("MSE for training: ", mean_squared_error(Y_train, svm_train))
print("MSE for testing: ", mean_squared_error(Y_test, svm_test))
print("R2 score for training: ", r2_score(Y_train, svm_train))
print("R2 score for testing: ", r2_score(Y_test, svm_test))


# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, svm_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("LGBM Regression - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, svm_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("LGBM Regression - Testing Data Scatter Plot")
plt.show()


##### Randomized Random Forest Regression algorithm #####

param = [
    {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 6],
        "max_leaf_nodes": [15, 20, 25],
    },
]

rf = RandomForestRegressor()
rs_rf = RandomizedSearchCV(rf, param, cv=2, n_jobs=-1, verbose=1)
rs_rf.fit(X_train, Y_train)
rs_rf_train = rs_rf.predict(X_train)
rs_rf_test = rs_rf.predict(X_test)

print("Randomized RandomForestRegressor Metrics:")
print("MSE for training: ", mean_squared_error(Y_train, rs_rf_train))
print("MSE for testing: ", mean_squared_error(Y_test, rs_rf_test))
print("R2 score for training: ", r2_score(Y_train, rs_rf_train))
print("R2 score for testing: ", r2_score(Y_test, rs_rf_test))

# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, rs_rf_train)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Randomized RandomForestRegressor - Training Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, rs_rf_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Randomized RandomForestRegressor - Testing Data Scatter Plot")
plt.show()

# Plotting the scatter plot of predicted vs true values for both training and testing sets
plt.figure(figsize=(8, 6))
plt.scatter(Y_train, rs_rf_train, alpha=0.3, label="Training")
plt.scatter(Y_test, rs_rf_test, alpha=0.3, label="Testing")
plt.plot([0, 10], [0, 10], linestyle="--", color="k", label="Perfect prediction")
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Randomized RandomForestRegressor - Training and Testing Sets Scatter Plot")
plt.legend()
plt.show()

# Plotting the residual plot for testing data
plt.scatter(rs_rf_test, rs_rf_test - Y_test, c="g", s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=10)
plt.xlabel("Predicted Ratings")
plt.ylabel("Residuals")
plt.title("Randomized RandomForestRegressor - Testing Data Residual Plot")
plt.show()

##### GriSearch LogisticRegression classification algorithm #####
param = [
    {
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky"],
    },
]

logi = LogisticRegression()
gs_logi = GridSearchCV(logi, param, cv=2, n_jobs=-1, verbose=1)
gs_logi.fit(X_train, Y_train)
logi_train = gs_logi.predict(X_train)
logi_test = gs_logi.predict(X_test)

train_accuracy = accuracy_score(logi_train, Y_train)
test_accuracy = accuracy_score(logi_test, Y_test)
print("\nLogistic Regression Metrics:")
print("Accuracy for training: ", train_accuracy)
print("Accuracy for testing: ", test_accuracy)

# Plotting the accuracy plot
plt.plot(["Training", "Testing"], [train_accuracy, test_accuracy], marker="o")
plt.title("Logistic Regression Accuracy")
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.show()

# Plotting the confusion matrix
ConfusionMatrixDisplay.from_estimator(gs_logi, X_test, Y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

##### LogisticRegression classification algorithm #####

# Create an instance of Logistic Regression model
logi = LogisticRegression()

# Fit the model on the training data
logi.fit(X_train, Y_train)

# Make predictions on the training and testing data
logi_train = logi.predict(X_train)
logi_test = logi.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(logi_train, Y_train)
test_accuracy = accuracy_score(logi_test, Y_test)

# Print the logistic regression metrics
print("\nLogistic Regression Metrics:")
print("Accuracy for training: ", train_accuracy)
print("Accuracy for testing: ", test_accuracy)

# Plot the accuracy comparison between training and testing
plt.plot(["Training", "Testing"], [train_accuracy, test_accuracy], marker="o")
plt.title("Logistic Regression Accuracy")
plt.xlabel("Dataset")
plt.ylabel("Accuracy")
plt.show()


# Plotting the confusion matrix
cm = confusion_matrix(Y_test, logi_test, labels=logi.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logi.classes_).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Pie Chart
correct_predictions = sum(Y_test == logi_test)
incorrect_predictions = len(Y_test) - correct_predictions

labels = ["Correct Predictions", "Incorrect Predictions"]
sizes = [correct_predictions, incorrect_predictions]
colors = ["green", "red"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)

plt.axis("equal")  # Equal aspect ratio ensures a circular pie chart
plt.title("Prediction Accuracy")
plt.show()

# Bar Chart
prediction_column = "predicted_rating"
target_column = "actual_rating"

# Get the predicted ratings and actual ratings from the dataset
predicted_ratings = logi_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Count the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[actual_ratings == rounded_predictions], bins=bins
)

# Create the bar chart
plt.bar(range(1, 11), correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Bar Chart of Correct Predictions")

# Display the bar chart
plt.show()

# Line Chart
predicted_ratings = logi_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create a line chart for the number of correct predictions
x = range(1, 11)

plt.plot(x, correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Line Chart of Correct Predictions")
plt.show()

# Area Chart
predicted_ratings = logi_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create an area chart for the number of correct predictions
x = range(1, 11)

plt.fill_between(x, correct_predictions, color="blue", alpha=0.3)
plt.plot(x, correct_predictions, color="blue", linewidth=2)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Area Chart of Correct Predictions")

plt.show()

# Histogram
plt.hist(Y_test, bins="auto", alpha=0.5, color="blue", label="Actual")
plt.hist(logi_test, bins="auto", alpha=0.5, color="red", label="Predicted")

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Histogram")
plt.legend()
plt.grid(True)
plt.show()

##### Perceptron Model classification algorithm #####


mlpcls = MLPClassifier(
    hidden_layer_sizes=(30, 30), activation="relu", random_state=1, max_iter=300
).fit(X_train, Y_train)

mlpcls_train = mlpcls.predict(X_train)
mlpcls_test = mlpcls.predict(X_test)
print("\nMulti Layer Perceptron Metrics:")
print("Accuracy for training ", accuracy_score(mlpcls_train, Y_train))
print("Accuracy for testing ", accuracy_score(mlpcls_test, Y_test))

# Plotting the scatter plot of actual vs predicted values
plt.scatter(Y_test, mlpcls_test, color="blue", label="Predicted Ratings")
plt.scatter(Y_test, Y_test, color="red", label="Actual Ratings")
plt.title("Scatter Plot -- Actual vs Predicted values for Multi Layer Perceptron Model")
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.legend()
plt.show()

# Plotting the step plot of accuracy
plt.step(
    [0, 1],
    [accuracy_score(mlpcls_train, Y_train), accuracy_score(mlpcls_test, Y_test)],
    where="post",
)
plt.title("Step Plot -- Accuracy for Multi Layer Perceptron Model")
plt.xticks([0, 1], ["Training", "Testing"])
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.show()

# Plotting the Confusion matrix
cm = confusion_matrix(Y_test, mlpcls_test)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("MultilayerPerceptron - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Pie Chart
correct_predictions = sum(Y_test == mlpcls_test)
incorrect_predictions = len(Y_test) - correct_predictions

labels = ["Correct Predictions", "Incorrect Predictions"]
sizes = [correct_predictions, incorrect_predictions]
colors = ["green", "red"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)

plt.axis("equal")  # Equal aspect ratio ensures a circular pie chart
plt.title("Prediction Accuracy")
plt.show()

# Bar Chart
prediction_column = "predicted_rating"
target_column = "actual_rating"

# Get the predicted ratings and actual ratings from the dataset
predicted_ratings = mlpcls_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Count the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[actual_ratings == rounded_predictions], bins=bins
)

# Create the bar chart
plt.bar(range(1, 11), correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Bar Chart of Correct Predictions")

# Display the bar chart
plt.show()

# Line Chart
predicted_ratings = mlpcls_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create a line chart for the number of correct predictions
x = range(1, 11)

plt.plot(x, correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Line Chart of Correct Predictions")
plt.show()

# Area Chart
predicted_ratings = mlpcls_test
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create an area chart for the number of correct predictions
x = range(1, 11)

plt.fill_between(x, correct_predictions, color="blue", alpha=0.3)
plt.plot(x, correct_predictions, color="blue", linewidth=2)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Area Chart of Correct Predictions")
plt.show()

# Histogram
plt.hist(Y_test, bins="auto", alpha=0.5, color="blue", label="Actual")
plt.hist(mlpcls_test, bins="auto", alpha=0.5, color="red", label="Predicted")

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Histogram")
plt.legend()
plt.grid(True)
plt.show()

##### Decision Tree Classifier algorithm #####

dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)

train_acc = []
test_acc = []
print("\nDecisionTreeClassifier Metrics:\n")
for i in range(1, 11):
    dt.fit(X_train, Y_train)
    train_pred = dt.predict(X_train)
    test_pred = dt.predict(X_test)
    train_acc.append(accuracy_score(train_pred, Y_train))
    test_acc.append(accuracy_score(test_pred, Y_test))
    print(f"Epoch {i} Training Accuracy: {train_acc[-1]}")
    print(f"Epoch {i} Testing Accuracy: {test_acc[-1]}")

# Plotting accuracy vs epoch
epochs = range(1, 11)
plt.plot(epochs, train_acc, "bo-", label="Training Accuracy")
plt.plot(epochs, test_acc, "go-", label="Testing Accuracy")
plt.title("Decision Tree Classifier Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plotting the scatter plot of actual vs predicted values
plt.scatter(Y_test, test_pred, alpha=0.3)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Decision Tree Classifier - Testing Data Scatter Plot")
plt.show()

# Plotting the confusion matrix
cm = confusion_matrix(Y_test, test_pred, labels=dt.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logi.classes_).plot(
    cmap="Blues"
)
plt.title("Decision Tree Classifier - Confusion Matrix")
plt.show()

# Pie Chart
correct_predictions = sum(Y_test == test_pred)
incorrect_predictions = len(Y_test) - correct_predictions

labels = ["Correct Predictions", "Incorrect Predictions"]
sizes = [correct_predictions, incorrect_predictions]
colors = ["green", "red"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)

plt.axis("equal")  # Equal aspect ratio ensures a circular pie chart
plt.title("Prediction Accuracy")
plt.show()

# Bar Chart
prediction_column = "predicted_rating"
target_column = "actual_rating"

# Get the predicted ratings and actual ratings from the dataset
predicted_ratings = test_pred
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Count the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[actual_ratings == rounded_predictions], bins=bins
)

# Create the bar chart
plt.bar(range(1, 11), correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Bar Chart of Correct Predictions")

# Display the bar chart
plt.show()

# Line Chart
predicted_ratings = test_pred
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create a line chart for the number of correct predictions
x = range(1, 11)

plt.plot(x, correct_predictions)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Line Chart of Correct Predictions")
plt.show()

# Area Chart
predicted_ratings = test_pred
actual_ratings = Y_test

# Convert decimal predictions to discrete values (1-10)
rounded_predictions = np.round(predicted_ratings)

# Calculate the number of correct predictions falling into each range
bins = range(1, 12)
correct_predictions, _ = np.histogram(
    rounded_predictions[rounded_predictions == actual_ratings], bins=bins
)

# Create an area chart for the number of correct predictions
x = range(1, 11)

plt.fill_between(x, correct_predictions, color="blue", alpha=0.3)
plt.plot(x, correct_predictions, color="blue", linewidth=2)
plt.xlabel("Range")
plt.ylabel("Count of Correct Predictions")
plt.title("Area Chart of Correct Predictions")

plt.show()

# Histogram
plt.hist(Y_test, bins="auto", alpha=0.5, color="blue", label="Actual")
plt.hist(test_pred, bins="auto", alpha=0.5, color="red", label="Predicted")

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Actual vs Predicted Histogram")
plt.legend()
plt.grid(True)
plt.show()


##### Long Short-Term Memory algorithm #####

# Define the model
model = Sequential()
model.add(LSTM(32, input_shape=(3006, 1)))
model.add(Dense(1))

# Reshape the X_train data
X_train = X_train.values.reshape(129038, 1006, 1)

# Reshape the y_train data
Y_train = Y_train.values.reshape(129038, 1)

# Reshape the Y_test data
Y_test = Y_test.values.reshape(53766, 1)

# Reshape the X_test data
X_test = X_test.values.reshape(53766, 1006, 1)

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Fit the model
model.fit(X_train, Y_train, epochs=10)

# Evaluate the model
model.evaluate(X_test, Y_test)

# Make predictions
predictions = model.predict(X_test)

mse = np.mean(np.square(predictions - Y_test))
print("Mean Squared Error (MSE):", mse)

mae = np.mean(np.abs(predictions - Y_test))
print("Mean Absolute Error (MAE):", mae)

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Plotting the scatter plot of predicted vs actual values for training data


# Make predictions on training data
train_predictions = model.predict(X_train)

# Reshape the predictions
train_predictions = train_predictions.reshape(train_predictions.shape[0])

# Create a scatter plot
plt.scatter(Y_train, train_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot: Predicted vs Actual (Training Data)")
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data

# Make predictions on testing data
test_predictions = model.predict(X_test)

# Reshape the predictions
test_predictions = test_predictions.reshape(test_predictions.shape[0])

# Create a scatter plot
plt.scatter(Y_test, test_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot: Predicted vs Actual (Testing Data)")
plt.show()

### TEXT PREPOCESSING , CREATION OF WORDCLOUDS ON THE REVIEW COLUMN , TEXT CLASSIFICATION (FEATURE EXTRACTION- BoW) , XGBoost MODEL ###


# CHECKING FOR NULL VALUES , DUPLICATE VALUES ,DROPPING UNNAMED COLUMNS
train_df.isnull().sum()
train_df = train_df.dropna(subset=["condition"])
train_df.isnull().sum()

train_df.duplicated().sum()
train_df.head()


# TEXT PREPROCESSING

# LOWER CASE
# STRING PUNCTUATIONS
# TOKENIZATION
# STEMMING
# all of this would be  done on the 'Reviews' column

train_df["review"] = train_df["review"].str.lower()
import string

string.punctuation
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
train_df["review"] = train_df["review"].str.replace(
    "[{}]".format(string.punctuation), ""
)
import nltk

train_df["review"] = train_df["review"].apply(nltk.word_tokenize)
train_df.head()

# Creating WordClouds for REVIEWS having rating >=5 and <=5
train_df["review"] = train_df["review"].apply(lambda x: " ".join(x))
from wordcloud import WordCloud

# Create an instance of WordCloud with specified parameters
wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")

# Generate word cloud for reviews with rating >= 5
rev5_wc = wc.generate(train_df[train_df["rating"] >= 5]["review"].str.cat(sep=" "))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(rev5_wc, interpolation="bilinear")
plt.axis("on")
plt.show()

# Generate word cloud for reviews with rating <= 5
rev4_wc = wc.generate(train_df[train_df["rating"] <= 5]["review"].str.cat(sep=" "))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(rev4_wc, interpolation="bilinear")
plt.axis("on")
plt.show()


# TEXT CLASSIFICATION- FEATURE SELECTION
# APPLYING BAGofWORDS feature on the processed Review

from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer
reviews = train_df["review"]
vectorizer = CountVectorizer(max_features=1000)
X_bow = vectorizer.fit_transform(reviews)


# Fit and transform the reviews into a BoW feature matrix
X_bow = vectorizer.fit_transform(reviews)

# Constructing the XGBoost Model
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Assign feature matrix X and target variable y
X = X_bow
y = train_df["rating"]

# Get unique labels in the target variable
unique_labels = y.unique()
print(unique_labels)

# Subtract 1 from y and convert it to integer type
y = y - 1
y = y.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Create an instance of the XGBoost classifier model
model = xgb.XGBClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision score
from sklearn.metrics import precision_score

print("Precision:", precision_score(y_test, y_pred, average="macro"))


# Plotting the graph to check for accuracy and precision .
accuracy = 0.5853173830027666
precision = 0.5977667150507872

metrics = ["accuracy", "precision"]
scores = [accuracy, precision]

x_pos = np.arange(len(metrics))

plt.bar(x_pos, scores, align="center", alpha=0.8)
plt.xticks(x_pos, metrics)
plt.ylabel("Score")
plt.title("Accuracy and Precision")

# Add labels to each bar
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, str(score), ha="center")

plt.show()

# Scatter Plot

# Make predictions on testing data
test_predictions = model.predict(X_test)

# Reshape the predictions
test_predictions = test_predictions.reshape(test_predictions.shape[0])

# Create a scatter plot
plt.scatter(y_test, test_predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter Plot: Predicted vs Actual (Testing Data)")
plt.show()


# Hyper parameter tuning using optuna


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.25, 0.99),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 69),
    }

    model = cbt.CatBoostRegressor(**params, silent=True)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(Y_test, predictions, squared=False)
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# CatBoost with hyper-parameters found using Optuna

cb_rge_1 = cbt.CatBoostRegressor(
    iterations=9458,
    learning_rate=0.0563603538149542,
    depth=8,
    subsample=0.7522145960722497,
    colsample_bylevel=0.9788529170933132,
    min_data_in_leaf=69,
)


cb_rge_1.fit(X_train, Y_train)

cb_preds = cb_rge_1.predict(X_test)
cb_pred1 = cb_rge_1.predict(X_train)

print("MSE for training: ", mean_squared_error(Y_train, cb_pred1))
print("MSE for testing: ", mean_squared_error(Y_test, cb_preds))
print("R2 score for training: ", r2_score(Y_train, cb_pred1))
print("R2 score for testing: ", r2_score(Y_test, cb_preds))

# scatter plot

# testing data
plt.scatter(Y_test, cb_preds)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("CatBoost Scatter Plot: Predicted vs Actual (Testing Data)")
plt.show()

# training data
plt.scatter(Y_train, cb_pred1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("CatBoost Scatter Plot: Predicted vs Actual (Training Data)")
plt.show()


## Creating labels
new = train_df["rating"]
labels = -1 * (new <= 4) + 1 * (new >= 7)
train_df["label"] = labels

## Check ratings to labels
train_df.plot(x="rating", y="label", kind="scatter")
plt.show()

## Creating a new column review_length
train_df["review_length"] = train_df["review"].apply(len)
train_df["review_length"].describe()

## Creating a plot for distribution of review lengths
train_df.hist("review_length", bins=np.arange(0, 1500, 100))
plt.title("Distribution of review lengths")
plt.xlabel("Review length")
plt.ylabel("Count")
plt.show()

## Converting reviews to padding sequences
WORDS = 1000
LENGTH = 100
N = 10000
DEPTH = 32

samples = train_df["review"].iloc[:N]
tokenizer = Tokenizer(num_words=WORDS)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
data_train = pad_sequences(sequences, maxlen=LENGTH)

## Converting labels to one-hot-categorical values
one_hot_labels = to_categorical(labels[:N], num_classes=3)

## Checking the shape
data_train.shape, one_hot_labels.shape


## Helper functions
def plot_history(history):
    fs, ax = plt.subplots(1, 2, figsize=(16, 7))

    accuracy = history.history["acc"]
    val_accuracy = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(accuracy) + 1)

    plt.sca(ax[0])
    plt.plot(epochs, accuracy, "bo", label="Training Accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.sca(ax[1])
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()

    plt.show()


## Creating a helper function for model training
def train_model(model, x, y, e=12, bs=32, v=1, vs=0.25):
    m = model.fit(x, y, epochs=e, batch_size=bs, verbose=v, validation_split=vs)
    return m


## First type - Embedding and Flatten
m1 = Sequential()
m1.add(Embedding(WORDS, DEPTH, input_length=LENGTH))
m1.add(Flatten())
m1.add(Dense(32, activation="relu"))
m1.add(Dense(3, activation="softmax"))
m1.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
m1.summary()

# Train the first type and plot the history
h1 = train_model(m1, data_train, one_hot_labels)
plot_history(h1)

# Second type - Embedding and LSTM
m2 = Sequential()
m2.add(Embedding(WORDS, DEPTH, input_length=LENGTH))
m2.add(LSTM(DEPTH))
m2.add(Dense(3, activation="softmax"))
m2.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
m2.summary()

# Train the second type and plot the history
h2 = train_model(m2, data_train, one_hot_labels)
plot_history(h2)

## Third type - Embedding and GRU
m3 = Sequential()
m3.add(Embedding(WORDS, DEPTH, input_length=LENGTH))
m3.add(GRU(LENGTH))
m3.add(Dense(3, activation="softmax"))
m3.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
m3.summary()

## Train the third type and plot the history
h3 = train_model(m3, data_train, one_hot_labels)
plot_history(h3)

## Fourth type - Embedding with Conv1D & MaxPooling1D
m4 = Sequential()
m4.add(Embedding(WORDS, DEPTH, input_length=LENGTH))
m4.add(Conv1D(DEPTH, 7, activation="relu"))
m4.add(MaxPooling1D(5))
m4.add(Conv1D(DEPTH, 7, activation="relu"))
m4.add(GlobalMaxPooling1D())
m4.add(Dense(3, activation="softmax"))
m4.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
m4.summary()

## Train the fourth type and plot the history
h4 = train_model(m4, data_train, one_hot_labels)
plot_history(h4)

# Fifth type - Embedding with mixed Conv1D and GRU
m5 = Sequential()
m5.add(Embedding(WORDS, DEPTH, input_length=LENGTH))
m5.add(Conv1D(DEPTH, 5, activation="relu"))
m5.add(MaxPooling1D(5))
m5.add(Conv1D(DEPTH, 7, activation="relu"))
m5.add(GRU(DEPTH, dropout=0.1, recurrent_dropout=0.5))
m5.add(Dense(3, activation="softmax"))
m5.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
m5.summary()

## Train the fifth type and plot the history
h5 = train_model(m5, data_train, one_hot_labels)
plot_history(h5)

##### KNearest Neighbours Algorithm #####
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, Y_train)
train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)

train_accuracy = accuracy_score(train_pred, Y_train)
test_accuracy = accuracy_score(test_pred, Y_test)
print("\n KNearest Neighbour Metrics: \n")
print("Accuracy for training: ", train_accuracy)
print("Accuracy for testing: ", test_accuracy)

# Plotting the confusion matrix
ConfusionMatrixDisplay.from_estimator(classifier, X_test, Y_test)
plt.title("KNearest Neighbours Confusion Matrix")
plt.show()


# Arima implementation
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model
model = ARIMA(Y_train, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions on the test data
arima_test = model_fit.forecast(steps=len(X_test))
arima_train = model_fit.forecast(steps=len(X_train))

# Evaluate the model
mse_train = mean_squared_error(Y_train, arima_train)
mse_test = mean_squared_error(Y_test, arima_test)
r2_test = r2_score(Y_test, arima_test)
r2_train = r2_score(Y_train, arima_train)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(Y_test, arima_test)

# Print evaluation metrics
print("ARIMA RESULTS")
print("Mean Squared train:", mse_train)
print("R-squared train:", r2_test)
print("Mean Squared test:", mse_test)
print("R-squared test:", r2_test)
print("rmse test:", rmse_test)
print("mae test:", mae_test)

plt.scatter(Y_train, arima_train)
plt.xlabel("Actual")
plt.ylabel("Predicted Arima")
plt.title("ARima Model: Training data Scatter plot")
# plt.show()
plt.scatter(Y_test, arima_test)
plt.xlabel("Actual")
plt.ylabel("Predicted Arima")
plt.title("ARima Model: Testing data Scatter plot")
plt.show()
# Plotting the scatter plot of predicted vs true values for both training and testing sets
plt.figure(figsize=(8, 6))
plt.scatter(Y_train, arima_train, alpha=0.3, label="Training")
plt.scatter(Y_test, arima_test, alpha=0.3, label="Testing")
plt.plot([0, 10], [0, 10], linestyle="--", color="k", label="Perfect prediction")
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.title("ARima model - Training and Testing Sets Scatter Plot")
plt.legend()
plt.show()
plt.scatter(arima_test, Y_test, c="g", s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=10)
plt.xlabel("Predicted Ratings")
plt.ylabel("Residuals")
plt.title("ARima Model - Testing Data Residual Plot")
plt.show()


### Sentiment Analysis using DL


# Function to seperate text and label
def load_dataset(file_path, num_samples):
    df = pd.read_csv(file_path, usecols=[3, 4], nrows=num_samples)
    df.columns = ["review", "rating"]

    text = df["review"].tolist()
    text = [str(t).encode("ascii", "replace") for t in text]
    text = np.array(text, dtype=object)[:]

    labels = df["rating"].tolist()
    labels = [1 if i >= 7 else 0 if i >= 5 else -1 for i in labels]
    labels = np.array(pd.get_dummies(labels), dtype=int)[:]

    return labels, text


# Split into train and test dataset
tmp_labels, tmp_text = load_dataset("datasets\drugsComTrain_raw.tsv", 568454)
test_labels, test_text = load_dataset("datasets\drugsComTest_raw.tsv", 500000)

# Make the model
hub_layer = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1",
    output_shape=[50],
    input_shape=[],
    dtype=tf.string,
    name="input",
    trainable=False,
)
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.summary()

# Fit the model
print("Training the model ...")
history = model.fit(
    test_text,
    test_labels,
    batch_size=128,
    epochs=50,
    verbose=1,
    validation_data=(test_text, test_labels),
)

# e.g. Predict some text
model.predict(["im feeling sick"])


# sarima
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(Y_train, order=(1, 1, 3), seasonal_order=(1, 1, 2, 12))
model_fit = model.fit()
print(model_fit.summary())
mean_actual = y.mean()
# Predict future values
forecasted_values1 = model_fit.predict(start=Y_test.index[0], end=Y_test.index[-1])
tss1 = ((Y_test - mean_actual) ** 2).sum()
rss1 = ((Y_test - forecasted_values1) ** 2).sum()
r_squared1 = 1 - (rss1 / tss1)
print("r2 value Sarima: ", r_squared1)
print("MAE: ", mean_absolute_error(Y_test, forecasted_values1))
print("MSE: ", mean_squared_error(Y_test, forecasted_values1))
print("RMSE: ", np.sqrt(mean_squared_error(Y_test, forecasted_values1)))
predictions = model_fit.predict()
predictions.plot()
plt.show()
