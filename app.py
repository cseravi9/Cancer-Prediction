from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model
with open('logistic_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Feature names used during training
prediction_feature = ["radius_mean", 'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

# Home route
@app.route('/')
def home():
    return render_template('index.html', features=prediction_feature)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # User input from the form
    user_input = [float(request.form[feature]) for feature in prediction_feature]
    
    # Scale user input
    user_input_scaled = StandardScaler().fit_transform([user_input])

    # Make a prediction
    prediction = loaded_model.predict(user_input_scaled)

    # Determine the diagnosis result
    result = "Malignant" if prediction[0] == 1 else "Benign"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
