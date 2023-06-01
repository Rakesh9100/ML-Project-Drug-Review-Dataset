# Performing Data Cleaning and EDA 

import pandas as pd, numpy as np
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, plot_confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt

## Reading the data
dtypes = { 'Unnamed: 0': 'int32', 'drugName': 'category', 'condition': 'category', 'review': 'category', 'rating': 'float16', 'date': 'category', 'usefulCount': 'int16' }
train_df = pd.read_csv('drugsComTrain_raw.tsv', sep='\t', dtype=dtypes)
# Randomly selecting 80% of the data from the training dataset
train_df = train_df.sample(frac=0.8, random_state=42)
test_df = pd.read_csv('drugsComTest_raw.tsv', sep='\t', dtype=dtypes)

## Converting date column to datetime format
train_df['date'], test_df['date'] = pd.to_datetime(train_df['date'], format='%B %d, %Y'), pd.to_datetime(test_df['date'], format='%B %d, %Y')

## Extracting day, month, and year into separate columns
for df in [train_df, test_df]:
    df['day'] = df['date'].dt.day.astype('int8')
    df['month'] = df['date'].dt.month.astype('int8')
    df['year'] = df['date'].dt.year.astype('int16')
    
    # CHECKING FOR NULL VALUES , DUPLICATE VALUES ,DROPPING UNNAMED COLUMNS 
    train_df.isnull().sum()
    train_df = train_df.dropna(subset=['condition'])
    train_df.isnull().sum()
    
 train_df.duplicated().sum()
 train_df.head()
 
 
 #TEXT PREPROCESSING

#LOWER CASE
#STRING PUNCTUATIONS
#TOKENIZATION
# STEMMING 
all of this would be  done on the 'Reviews' column

train_df['review']= train_df['review'].str.lower()
import  string
string.punctuation
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
train_df['review'] = train_df['review'].str.replace('[{}]'.format(string.punctuation), '')
import nltk

train_df['review'] = train_df['review'].apply(nltk.word_tokenize)
train_df.head()

#Creating WordClouds for REVIEWS having rating >=5 and <=5
train_df['review'] = train_df['review'].apply(lambda x: ' '.join(x))
from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')

rev5_wc=wc.generate(train_df[train_df['rating']>=5]['review'].str.cat(sep=" "))
plt.figure(figsize=(10, 5))
plt.imshow(rev5_wc, interpolation='bilinear')
plt.axis('on')
plt.show()

rev4_wc=wc.generate(train_df[train_df['rating']<=5]['review'].str.cat(sep=" "))
plt.figure(figsize=(10, 5))
plt.imshow(rev4_wc, interpolation='bilinear')
plt.axis('on')
plt.show()

# TEXT CLASSIFICATION- FEATURE SELECTION 
# APPLYING BAGofWORDS feature on the processed Review 

from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer

vectorizer = CountVectorizer(max_features=1000)
X_bow = vectorizer.fit_transform(reviews)


# Fit and transform the reviews into a BoW feature matrix
X_bow= vectorizer.fit_transform(reviews)

#Constructing the XGBoost Model 
import xgboost as xgb
from sklearn.model_selection import train_test_split
X=X_bow
y=train_df['rating']
unique_labels = y.unique()
print(unique_labels)
y=y-1
y=y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import precision_score
print("Precision:",precision_score(y_test,y_pred,average='macro'))

#Plotting the graph to check for accuracy and precision .
accuracy= 0.5853173830027666
precision=0.5977667150507872

metrics=['accuracy','precision']
scores=[accuracy,precision]

x_pos=np.arange(len(metrics))

plt.bar(x_pos,scores,align='center',alpha=0.8)
plt.xticks(x_pos, metrics)
plt.ylabel('Score')
plt.title('Accuracy and Precision')

# Add labels to each bar
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, str(score), ha='center')

plt.show()

#Scatter Plot 

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








    
