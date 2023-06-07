import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.linear_model import LinearRegression
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from gensim.models import Word2Vec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, plot_confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Download the necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 

## Reading the data
dtypes = { 'Unnamed: 0': 'int32', 'drugName': 'category', 'condition': 'category', 'review': 'category', 'rating': 'float16', 'date': 'category', 'usefulCount': 'int16' }
train_df = pd.read_csv('drugsComTrain_raw.tsv',delimiter='\t', dtype=dtypes)
test_df= pd.read_csv('drugsComTest_raw.tsv',delimiter='\t', dtype=dtypes)
# Randomly selecting 80% of the data from the training dataset
total_samples = len(train_df)  # or total number of records
samples_to_read = int(0.8 * total_samples)  # 50% of the total
train_df = train_df.sample(n=samples_to_read, random_state=42)


## Converting date column to datetime format
train_df['date'], test_df['date'] = pd.to_datetime(train_df['date'], format='%B %d, %Y'), pd.to_datetime(test_df['date'], format='%B %d, %Y')

## Extracting day, month, and year into separate columns
for df in [train_df, test_df]:
    df['day'] = df['date'].dt.day.astype('int8')
    df['month'] = df['date'].dt.month.astype('int8')
    df['year'] = df['date'].dt.year.astype('int16')

## Suppressing MarkupResemblesLocatorWarning, FutureWarning and ConvergenceWarning
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

## Defining function to decode HTML-encoded characters
def decode_html(text):
    decoded_text = BeautifulSoup(text, 'html.parser').get_text()
    return decoded_text

## Applying the function to the review column
train_df['review'], test_df['review'] = train_df['review'].apply(decode_html), test_df['review'].apply(decode_html)

## Dropped the original date column and removed the useless column
train_df, test_df = [df.drop('date', axis=1).drop(df.columns[0], axis=1) for df in (train_df, test_df)]

## Handling the missing values
train_imp, test_imp = [pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df)) for df in (train_df, test_df)]

## Assigning old column names
train_imp.columns = ['drugName', 'condition', 'review', 'rating', 'usefulCount', 'day', 'month', 'year']
test_imp.columns = ['drugName', 'condition', 'review', 'rating', 'usefulCount', 'day', 'month', 'year']

## Converting the text in the review column to numerical data
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
train_reviews_tfid = vectorizer.fit_transform(train_imp['review'])
test_reviews_tfid = vectorizer.transform(test_imp['review'])

# word2vec vectorizer
# Initialize stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()

column_name = 'review'
# pre process
def preprocess_text(text): 
    
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char not in punctuation and not char.isdigit()])
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # stemming or lemmatization
    stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Using stemming
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(stemmed_tokens)  # Or use lemmatized_tokens if using lemmatization
    
    return preprocessed_text

# Preprocess the review column
train_imp[column_name] = train_imp[column_name].apply(preprocess_text)
test_imp[column_name] = test_imp[column_name].apply(preprocess_text)

# training on Word2Vec Model
model = Word2Vec(train_df[column_name], min_count=1)

# Extract review embeddings from train data
dropped_indices = []  # List to store the indices of dropped ratings
review_embeddings = []
for idx,review in enumerate(train_df[column_name]):
    embedding = []
    for word in review:
        if word in model.wv:
            embedding.append(model.wv[word]) 
    if len(embedding)>0:
        review_embeddings.append(sum(embedding) / len(embedding))
    else:
        review_embeddings.append([])  # Assigning an empty list
        dropped_indices.append(idx)

# Extract review embeddings from test data
test_dropped_indices = []  
test_review_embeddings = []
for idx,review in enumerate(test_df[column_name]):
    test_embedding = []
    for word in review:
        if word in model.wv:
            test_embedding.append(model.wv[word]) 
    if len(test_embedding)>0:
        test_review_embeddings.append(sum(test_embedding) / len(test_embedding))
    else:
        test_review_embeddings.append([])  
        test_dropped_indices.append(idx)


review_lengths = max([len(review) for review in review_embeddings])
# Pad sequences to a fixed length
padded_embeddings = pad_sequences(review_embeddings, padding='post', truncating='post', maxlen=100) #max_length
test_padded_embeddings = pad_sequences(test_review_embeddings, padding='post', truncating='post', maxlen=100) 

#Dimensionality reduction using PCA for word2vec and using Truncated SVD for Tfid
desired_num_features = 100

# Reduce Word2Vec embeddings
pca_word2vec = PCA(n_components=desired_num_features)
reduced_word2vec_embeddings = pca_word2vec.fit_transform(padded_embeddings)
test_reduced_word2vec_embeddings = pca_word2vec.fit_transform(test_padded_embeddings)



# Reduce TF-IDF features
svd_tfidf = TruncatedSVD(n_components=100)
reduced_tfidf_features = svd_tfidf.fit_transform(train_reviews_tfid)
test_reduced_tfidf_features = svd_tfidf.fit_transform(test_reviews_tfid)

# merging the embeddings in a dataframe
train_imp = pd.concat([train_imp, pd.DataFrame(reduced_tfidf_features), pd.DataFrame(reduced_word2vec_embeddings)], axis=1)
test_imp = pd.concat([test_imp, pd.DataFrame(test_reduced_tfidf_features), pd.DataFrame(test_reduced_word2vec_embeddings)], axis=1)


## Replacing the review column with the numerical data
train_imp.drop('review', axis=1, inplace=True)
test_imp.drop('review', axis=1, inplace=True)

## Encoding the categorical columns
for i in ["drugName", "condition"]:
    train_imp[i] = LabelEncoder().fit_transform(train_imp[i])
    test_imp[i] = LabelEncoder().fit_transform(test_imp[i])

## Converting the data types of columns to reduce the memory usage
train_imp, test_imp = train_imp.astype('float16'), test_imp.astype('float16')
train_imp[['drugName', 'condition', 'usefulCount', 'year']] = train_imp[['drugName', 'condition', 'usefulCount', 'year']].astype('int16')
test_imp[['drugName', 'condition', 'usefulCount', 'year']] = test_imp[['drugName', 'condition', 'usefulCount', 'year']].astype('int16')
train_imp[['rating']] = train_imp[['rating']].astype('float16')
test_imp[['rating']] = test_imp[['rating']].astype('float16')
train_imp[['day', 'month']] = train_imp[['day', 'month']].astype('int8')
test_imp[['day', 'month']] = test_imp[['day', 'month']].astype('int8')

#print(train_imp.iloc[:,:15].dtypes)
#print(test_imp.iloc[:,:15].dtypes)

## Splitting the train and test datasets into feature variables
X_train, Y_train = train_imp.drop('rating', axis=1), train_imp['rating']
X_test, Y_test = test_imp.drop('rating', axis=1), test_imp['rating']

##### LinearRegression regression algorithm #####

linear=LinearRegression()
linear.fit(X_train, Y_train)
line_train=linear.predict(X_train)
line_test=linear.predict(X_test)

print("Linear Regression Metrics:")
print("MSE for training: ", mean_squared_error(Y_train, line_train))
print("MSE for testing: ", mean_squared_error(Y_test, line_test))
print("R2 score for training: ", r2_score(Y_train, line_train))
print("R2 score for testing: ", r2_score(Y_test, line_test))

# Plotting the scatter plot of predicted vs actual values for training data
plt.scatter(Y_train, line_train)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Linear Regression - Training Data Scatter Plot')
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, line_test)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Linear Regression - Testing Data Scatter Plot')
plt.show()

# Plotting the scatter plot of predicted vs true values for both training and testing sets
plt.figure(figsize=(8,6))
plt.scatter(Y_train, line_train, alpha=0.3, label='Training')
plt.scatter(Y_test, line_test, alpha=0.3, label='Testing')
plt.plot([0,10], [0,10], linestyle='--', color='k', label='Perfect prediction')
plt.xlabel('True Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Linear regression - Training and Testing Sets Scatter Plot')
plt.legend()
plt.show()

# Plotting the residual plot for testing data
plt.scatter(line_test, line_test - Y_test, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=10)
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.title('Linear Regression - Testing Data Residual Plot')
plt.show()

##### LogisticRegression classification algorithm #####

logi=LogisticRegression()
logi.fit(X_train, Y_train)
logi_train=logi.predict(X_train)
logi_test=logi.predict(X_test)

train_accuracy = accuracy_score(logi_train, Y_train)
test_accuracy = accuracy_score(logi_test, Y_test)
print("\nLogistic Regression Metrics:")
print("Accuracy for training: ", train_accuracy)
print("Accuracy for testing: ", test_accuracy)

# Plotting the accuracy plot
plt.plot(['Training', 'Testing'], [train_accuracy, test_accuracy], marker='o')
plt.title('Logistic Regression Accuracy')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.show()

# Plotting the confusion matrix
plot_confusion_matrix(logi, X_test, Y_test)
plt.title('Logistic Regression Confusion Matrix')
plt.show()

##### Perceptron Model classification algorithm #####

perce = Perceptron(max_iter=1000, eta0=0.5)
perce.fit(X_train, Y_train)

perce_train=perce.predict(X_train)
perce_test=perce.predict(X_test)
print("\nPerceptron Metrics:")
print("Accuracy for training ",accuracy_score(perce_train, Y_train))
print("Accuracy for testing ",accuracy_score(perce_test, Y_test))

# Plotting the scatter plot of actual vs predicted values
plt.scatter(Y_test, perce_test, color='blue', label='Predicted Ratings')
plt.scatter(Y_test, Y_test, color='red', label='Actual Ratings')
plt.title('Scatter Plot -- Actual vs Predicted values for Perceptron Model')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.legend()
plt.show()

# Plotting the step plot of accuracy
plt.step([0, 1], [accuracy_score(perce_train, Y_train), accuracy_score(perce_test, Y_test)], where='post')
plt.title('Step Plot -- Accuracy for Perceptron Model')
plt.xticks([0, 1], ['Training', 'Testing'])
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()

# Plotting the Confusion matrix
cm = confusion_matrix(Y_test, perce_test)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Perceptron - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

##### Decision Tree regression algorithm #####

dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)

train_acc = []
test_acc = []
print("\nDecisionTreeClassifier Metrics:\n")
for i in range(1, 11):
    dt.fit(X_train, Y_train)
    train_pred=dt.predict(X_train)
    test_pred=dt.predict(X_test)
    train_acc.append(accuracy_score(train_pred, Y_train))
    test_acc.append(accuracy_score(test_pred, Y_test))
    print(f"Epoch {i} Training Accuracy: {train_acc[-1]}")
    print(f"Epoch {i} Testing Accuracy: {test_acc[-1]}")

# Plotting accuracy vs epoch
epochs = range(1, 11)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, test_acc, 'go-', label='Testing Accuracy')
plt.title('Decision Tree Classifier Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the scatter plot of actual vs predicted values
plt.scatter(Y_test, test_pred, alpha=0.3)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Decision Tree Classifier - Testing Data Scatter Plot')
plt.show()

# Plotting the confusion matrix
cm = confusion_matrix(Y_test, test_pred)
disp = plot_confusion_matrix(dt, X_test, Y_test, cmap=plt.cm.Blues)
disp.ax_.set_title('Decision Tree Classifier - Confusion Matrix')
plt.show()
