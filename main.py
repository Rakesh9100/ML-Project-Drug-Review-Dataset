import pandas as pd
import numpy as np
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
import catboost as cbt
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import gensim
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.parsing.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

## Reading the data
dtypes = {
    "Unnamed: 0": "int32",
    "drugName": "category",
    "condition": "category",
    "review": "category",
    "rating": "float16",
    "date": "string",
    "usefulCount": "int16",
}
train_df = pd.read_csv(
    r"datasets\drugsComTrain_raw.tsv", sep="\t", quoting=2, dtype=dtypes
)
train_df = train_df.sample(frac=0.8, random_state=42)
test_df = pd.read_csv(
    r"datasets\drugsComTest_raw.tsv", sep="\t", quoting=2, dtype=dtypes
)

## Converting date column to datetime format
train_df["date"], test_df["date"] = pd.to_datetime(
    train_df["date"], format="%B %d, %Y"
), pd.to_datetime(test_df["date"], format="%B %d, %Y")

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

## Assigning old column names
train_imp.columns = train_df.columns
test_imp.columns = test_df.columns

## Tokenization and stemming
p_stemmer = PorterStemmer()


def tokenize_and_stem(text):
    tokens = gensim.utils.simple_preprocess(text)
    stems = [p_stemmer.stem(token) for token in tokens]
    return stems


## Creating Word2Vec embeddings
combined_text = train_imp["review"].append(test_imp["review"])
tokenized_stemmed_text = combined_text.map(tokenize_and_stem)
word2vec_model = gensim.models.Word2Vec(
    tokenized_stemmed_text,
    size=100,
    window=5,
    min_count=1,
    workers=4,
    iter=100,
)

## Saving Word2Vec vectors for training and test data
train_word2vec = pd.DataFrame(
    np.array(
        [
            np.mean([word2vec_model.wv[word] for word in words], axis=0)
            for words in tokenized_stemmed_text[:train_imp.shape[0]]
        ]
    )
)
train_word2vec.to_csv("train_word2vec.csv", index=False)

test_word2vec = pd.DataFrame(
    np.array(
        [
            np.mean([word2vec_model.wv[word] for word in words], axis=0)
            for words in tokenized_stemmed_text[train_imp.shape[0] :]
        ]
    )
)
test_word2vec.to_csv("test_word2vec.csv", index=False)

## Joining Word2Vec vectors with original dataframes
train_df = pd.concat([train_imp, train_word2vec], axis=1)
test_df = pd.concat([test_imp, test_word2vec], axis=1)

## Dropping unnecessary columns
train_df = train_df.drop(["review"], axis=1)
test_df = test_df.drop(["review"], axis=1)

## Encoding categorical columns
encoder = LabelEncoder()
for col in ["drugName", "condition"]:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

## Converting column types to reduce memory usage
train_df = train_df.astype(
    {"drugName": "int16", "condition": "int16", "rating": "float16",}
)
test_df = test_df.astype(
    {"drugName": "int16", "condition": "int16", "rating": "float16",}
)

## Splitting the data into feature variables (X) and target variable (Y)
X_train, Y_train = train_df.drop("rating", axis=1), train_df["rating"]
X_test, Y_test = test_df.drop("rating", axis=1), test_df["rating"]

## Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x=Y_train)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

## Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

## Making predictions
linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)

## Evaluating Linear Regression
mse_train = mean_squared_error(Y_train, linear_train_pred)
mse_test = mean_squared_error(Y_test, linear_test_pred)
r2_train = r2_score(Y_train, linear_train_pred)
r2_test = r2_score(Y_test, linear_test_pred)

print("Linear Regression Results:")
print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Train R2 Score: {r2_train:.4f}")
print(f"Test R2 Score: {r2}")