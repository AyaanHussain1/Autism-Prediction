import numpy as np
import pandas as pd
import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

new_data = {
    'A1_Score': 1, 'A2_Score': 1, 'A3_Score': 1, 'A4_Score': 1, 'A5_Score': 1,
    'A6_Score': 1, 'A7_Score': 1, 'A8_Score': 1, 'A9_Score': 1, 'A10_Score': 1,
    'age': 25,
    'gender': 'm',
    'ethnicity': 'White-European',
    'jaundice': 'no',
    'austim': 'yes',
    'contry_of_res': 'United States',
    'used_app_before': 'no',
    'result': 14.5,
    'relation': 'Self'
}

#  Convert to a DataFrame row
input_df = pd.DataFrame([new_data])

# exact column order model expects
feature_order = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 
    'age', 'gender', 'ethnicity', 'jaundice', 'austim', 
    'contry_of_res', 'used_app_before', 'result', 'relation'
]
input_df = input_df[feature_order]

for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

prediction = model.predict(input_df)[0]

if prediction == 1:
    print("Prediction: Positive for Autism Traits")
else:
    print("Prediction: Negative for Austism Traits")