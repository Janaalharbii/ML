import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the data
df = pd.read_csv('/content/drive/MyDrive/trainingEVC/week2/DAY4/insurance.csv')

# Create the Streamlit UI
st.title("Insurance Charges Predictor")

# User input fields
age = st.number_input("Age", min_value=18, max_value=64, value=30, step=1)
sex = st.selectbox("Sex", options=["male", "female"])
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, step=1)
smoker = st.selectbox("Smoker", options=["yes", "no"])

# Preprocess the data
X = df[['age', 'children']]
one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(df[['sex', 'smoker']]).toarray()
X = pd.concat([X, pd.DataFrame(X_encoded, columns=one_hot_encoder.get_feature_names_out(['sex', 'smoker']))], axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make a prediction based on the user's input
user_input = np.array([[age, children, 
                       1 if sex == 'male' else 0, 1 if sex == 'female' else 0,
                       1 if smoker == 'yes' else 0, 1 if smoker == 'no' else 0]])
predicted_charges = model.predict(user_input)[0]

# Display the predicted charges
st.write(f"The predicted insurance charges for the given input are: ${predicted_charges:.2f}")