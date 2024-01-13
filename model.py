import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Add pickle for model saving

# Function for model building
def model_building(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, predictions)
    return score, accuracy, predictions

# Function for confusion matrix visualization
def cm_metrix_graph(cm):
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

# Function for cross-validation scoring
def cross_val_scorring(model):
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print("\nCross-Validation Scores for", str(type(model).__name__), ":", scores)
    print("Mean CV Score:", np.mean(scores))

# Function for hyperparameter tuning using GridSearchCV
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("\nGrid Search Results for", str(type(model).__name__), ":")
    print("Best Parameters:", grid_search.best_params_)
    print("Best Estimator:", grid_search.best_estimator_)
    print("Best Score:", grid_search.best_score_)

# Load data
data = pd.read_csv("data.csv")

# Explore data
data.info()
data = data.dropna(axis='columns')
data.describe(include="O")
data.diagnosis.value_counts()

# Data Filtering
labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)

# Feature Selection
prediction_feature = ["radius_mean", 'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']
targeted_feature = 'diagnosis'
X = data[prediction_feature]
y = data.diagnosis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Model Implementation
models_list = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC": SVC(),
}

df_prediction = []
confusion_matrixs = []
df_prediction_cols = ['model_name', 'score', 'accuracy_score', "accuracy_percentage"]

for name, model in models_list.items():
    (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test)
    print("\n\nClassification Report of '"+str(name), "'\n")
    print(classification_report(y_test, predictions))
    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])
    confusion_matrixs.append(confusion_matrix(y_test, predictions))

df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)

# Visualize Confusion Matrices
# plt.figure(figsize=(10, 2))
# for cm in confusion_matrixs:
#     cm_metrix_graph(cm)
#     plt.tight_layout(pad=True)

# Cross-Validation
for name, model in models_list.items():
    cross_val_scorring(model)

# Grid Search for Hyperparameter Tuning
tune_model(LogisticRegression(), {'C': [0.1, 1, 10, 100]}, X_train, y_train)
tune_model(KNeighborsClassifier(), {'n_neighbors': list(range(1, 30))}, X_train, y_train)
tune_model(SVC(), {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, X_train, y_train)
tune_model(RandomForestClassifier(), {'n_estimators': [200, 400]}, X_train, y_train)

# Save a trained model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
filename = 'logistic_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(logistic_model, file)

# User Input for Prediction
print("\nEnter the values for prediction:")
user_input = []
for feature in prediction_feature:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Scale user input
user_input = sc.transform([user_input])

# Load the saved model and make a prediction
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

prediction = loaded_model.predict(user_input)
print("\nPrediction:", "Malignant" if prediction[0] == 1 else "Benign")
