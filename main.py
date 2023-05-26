import pandas as pd, numpy as np  
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

## Reading the data
dtypes = { 'Unnamed: 0': 'int32', 'drugName': 'category', 'condition': 'category', 'review': 'category', 'rating': 'float16', 'date': 'string', 'usefulCount': 'int16' }
train_df = pd.read_csv(r"datasets\drugsComTrain_raw.tsv", sep='\t', quoting=2, dtype=dtypes, parse_dates=['date'])

train_df = train_df.sample(frac=0.5, random_state=42)
test_df = pd.read_csv(r"datasets\drugsComTest_raw.tsv", sep='\t', quoting=2, dtype=dtypes, parse_dates=['date'])

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

## Handling the missing values and assigning old column names
train_imp, test_imp = [pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df), columns=df.columns) for df in (train_df, test_df)]

## Converting the text in the review column to numerical data
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
train_reviews = vectorizer.fit_transform(train_imp['review'])
test_reviews = vectorizer.transform(test_imp['review'])

## Replacing the review column with the numerical data
train_imp.drop('review', axis=1, inplace=True)
test_imp.drop('review', axis=1, inplace=True)
train_imp = pd.concat([train_imp, pd.DataFrame(train_reviews.toarray()).add_prefix('review')], axis=1)
test_imp = pd.concat([test_imp, pd.DataFrame(test_reviews.toarray()).add_prefix('review')], axis=1)

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


X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


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

##### Randomized Random Forest Regression algorithm #####

param = [
    {'n_estimators': [100, 200, 300], 
     'max_depth': [3, 4, 6], 
     'max_leaf_nodes': [15, 20, 25]}, 
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
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Randomized RandomForestRegressor - Training Data Scatter Plot')
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data
plt.scatter(Y_test, rs_rf_test)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Randomized RandomForestRegressor - Testing Data Scatter Plot')
plt.show()

# Plotting the scatter plot of predicted vs true values for both training and testing sets
plt.figure(figsize=(8,6))
plt.scatter(Y_train, rs_rf_train, alpha=0.3, label='Training')
plt.scatter(Y_test, rs_rf_test, alpha=0.3, label='Testing')
plt.plot([0,10], [0,10], linestyle='--', color='k', label='Perfect prediction')
plt.xlabel('True Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Randomized RandomForestRegressor - Training and Testing Sets Scatter Plot')
plt.legend()
plt.show()

# Plotting the residual plot for testing data
plt.scatter(rs_rf_test, rs_rf_test - Y_test, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=10)
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.title('Randomized RandomForestRegressor - Testing Data Residual Plot')
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

cm = confusion_matrix(Y_test, logi_test, labels=logi.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logi.classes_).plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()


##### Perceptron Model classification algorithm #####


mlpcls = MLPClassifier(hidden_layer_sizes=(30,30),activation="relu",random_state=1,max_iter=300).fit(X_train, Y_train)

mlpcls_train=mlpcls.predict(X_train)
mlpcls_test=mlpcls.predict(X_test)
print("\nMulti Layer Perceptron Metrics:")
print("Accuracy for training ",accuracy_score(mlpcls_train,Y_train))
print("Accuracy for testing ",accuracy_score(mlpcls_test,Y_test))


# Plotting the scatter plot of actual vs predicted values
plt.scatter(Y_test, mlpcls_test, color='blue', label='Predicted Ratings')
plt.scatter(Y_test, Y_test, color='red', label='Actual Ratings')
plt.title('Scatter Plot -- Actual vs Predicted values for Multi Layer Perceptron Model')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.legend()
plt.show()

# Plotting the step plot of accuracy
plt.step([0, 1], [accuracy_score(mlpcls_train, Y_train), accuracy_score(mlpcls_test, Y_test)], where='post')
plt.title('Step Plot -- Accuracy for Multi Layer Perceptron Model')
plt.xticks([0, 1], ['Training', 'Testing'])
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()

# Plotting the Confusion matrix
cm = confusion_matrix(Y_test, mlpcls_test)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('MultilayerPerceptron - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

##### Decision Tree Classifier algorithm #####

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
cm = confusion_matrix(Y_test, test_pred, labels=dt.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logi.classes_).plot(cmap='Blues')
plt.title('Decision Tree Classifier - Confusion Matrix')
plt.show()


##### Long Short-Term Memory algorithm #####

# Define the model
model = Sequential()
model.add(LSTM(32, input_shape=(3006, 1)))
model.add(Dense(1))

# Reshape the X_train data
X_train = X_train.values.reshape(129038, 3006, 1)

# Reshape the y_train data
Y_train = Y_train.values.reshape(129038, 1)

#Reshape the Y_test data
Y_test = Y_test.values.reshape(53766,1)

#Reshape the X_test data
X_test = X_test.values.reshape(53766,3006,1)

# Compile the model
model.compile(loss='mse', optimizer='adam')

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
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot: Predicted vs Actual (Training Data)')
plt.show()

# Plotting the scatter plot of predicted vs actual values for testing data

# Make predictions on testing data
test_predictions = model.predict(X_test)

# Reshape the predictions
test_predictions = test_predictions.reshape(test_predictions.shape[0])

# Create a scatter plot
plt.scatter(Y_test, test_predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot: Predicted vs Actual (Testing Data)')
plt.show()