import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model and test data
model = joblib.load('XGBoost.pkl')
X_test = pd.read_csv('X_test.csv')

# Define feature names
feature_names = [
    "SLC6A13", "ANLN", "MARCO", "SYT13", "ARG2", "MEFV", "ZNF29P",
    "FLVCR2", "PTGFR", "CRISP2", "EME1", "IL22RA2", "SLC29A4",
    "CYBB", "LRRC25", "SCN8A", "LILRA6", "CTD_3080P12_3", "PECAM1"
]

st.title("Prediction of the Risk of Non-Small Cell Lung Cancer Based on the Expression Levels of Diabetes-Related Genes.")

# Create input widgets for each feature
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(feature, min_value=0, max_value=100000, value=161)

# Store the input values in a DataFrame with correct feature order
feature_values = [inputs[fn] for fn in feature_names]
feature_df = pd.DataFrame([feature_values], columns=feature_names)

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    # Get the predicted class and probabilities
    predicted_class = model.predict(feature_df)[0]
    predicted_proba = model.predict_proba(feature_df)[0]
    
    # Display the results
    st.write(f"**Predicted Class:** {predicted_class} (1: Tumor, 0: Normal)")
    st.write(f"**Predicted Probabilities:** {predicted_proba}")
    
    # Provide advice based on the predicted class
    if predicted_class == 1:
        advice = ("According to our model, we're sorry to tell you that you're at high risk of having non-small cell lung cancer. Please contact a professional doctor for a thorough check-up as soon as possible. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    else:
        advice = ("According to our model, we're glad to inform you that your risk of non-small cell lung cancer is low. But if you feel unwell, consult a professional doctor. Wish you good health. Note that our result isn't a final diagnosis. The specific result should be based on the diagnosis from a relevant hospital.")
    st.write(advice)
    
    # SHAP analysis
    st.subheader("SHAP Force Plot Explanation")
    
    # Create a SHAP explainer
    explainer_shap = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer_shap.shap_values(feature_df)
    
    # Get the correct class index and SHAP values
    class_idx = int(predicted_class)
    
    # Handle SHAP values structure
    if isinstance(shap_values, list):
        # For multi-class: shap_values is [class0_array, class1_array]
        # Each array shape: (n_samples, n_features)
        shap_value = shap_values[class_idx][0]  # Get first sample's SHAP values
    else:
        # For binary classification with one output
        shap_value = shap_values[0]  # Get first sample's SHAP values
    
    # Get expected value
    if isinstance(explainer_shap.expected_value, (list, np.ndarray)):
        expected_value = explainer_shap.expected_value[class_idx]
    else:
        expected_value = explainer_shap.expected_value
    
    # Create force plot
    plt.figure()
    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value,
        features=feature_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()