import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
#init_notebook_mode(connected=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
st.set_page_config(page_title="Hello", page_icon="👋")

st.set_option('deprecation.showPyplotGlobalUse', False)


st.write("# Welcome to SHAP Explainer! 👋")
st.markdown(
    """
    This app serves as a starter guide for understanding and explaining regression models using SHAP values.
    It contains the following sections:

    - **Feature Importance (📊)**: Analyze the significance of different features in the model.
    - **Regression Stats (📈)**: Get statistical summaries and evaluations for the model.
    - **Individual Predictions (👥)**: Generate and view individual predictions.
    - **Feature Dependence (🔍)**: Examine how different features interact within the model.
    - **What If (❓)**: Conduct 'What-If' analyses to understand how changes in feature values could affect predictions.

    ### Want to learn more about SHAP values?
    - Check out Christoph Molnar's [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shap.html) 
        book and his new guide on [Interpreting Machine Learning Models With SHAP](https://www.amazon.de/dp/B0CHL7W1DL)
    - Jump into [SHAP library](https://github.com/shap/shap/tree/master) repo on GitHub.

"""
)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

@st.cache_data
def load_data(file_path="./data/interaction_dataset_with_names.csv"):
    df = pd.read_csv(file_path,sep='\t')
    return df
if 'data' not in st.session_state:
    st.session_state.data = load_data()

@st.cache_data
def load_input_X(dataframe):
    X = dataframe.drop(['bonus','employee_name'], axis=1)
    return X
X = load_input_X(st.session_state.data)
if 'X' not in st.session_state:
    st.session_state.X = X

@st.cache_data
def load_input_y(dataframe):
    y = dataframe['bonus']
    return y
y = load_input_y(st.session_state.data)
if 'y' not in st.session_state:
    st.session_state.y = y

X_features = list(st.session_state.X.columns)
if 'X_features' not in st.session_state:
    st.session_state.X_features = X_features

@st.cache_resource
def load_trained_model(input_variables, target_variable):
    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(input_variables, target_variable)
    return model
model = load_trained_model(st.session_state.X, st.session_state.y)
if 'model' not in st.session_state:
    st.session_state.model = model

@st.cache_data
def get_predictions(_model, input_variables):
    predictions = _model.predict(input_variables)
    return predictions
y_pred = get_predictions(st.session_state.model, st.session_state.X)
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = y_pred

# Get SHAP values
@st.cache_resource
def get_shap_explainer(_model, input_variables):
    shap_explainer = shap.Explainer(_model,input_variables[:10])
    return shap_explainer
explainer = get_shap_explainer(st.session_state.model, st.session_state.X)
if 'explainer' not in st.session_state:
    st.session_state.explainer = explainer

@st.cache_data
def get_shap_values(_shap_explainer, input_variables):
    shap_vals = _shap_explainer(input_variables, check_additivity=False)
    return shap_vals
shap_values = get_shap_values(st.session_state.explainer, st.session_state.X)
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = shap_values


# st.header("Feature Descriptions")
# st.caption("A brief description of each feature")
# temp_df = pd.DataFrame(
# [        
# {"Feature": "experience", "Description": "Years of experience in the industry"},
# {"Feature": "degree", "Description": "Level of education"},
# {"Feature": "sales", "Description": "Sales performance"},
# {"Feature": "performance", "Description": "Overall performance"},
# {"Feature": "days_late", "Description": "Number of days late on projects"},
# ]
# )
# st.dataframe(temp_df, hide_index=True)

# st.header("SHAP Summary")
# st.subheader("Which features have the biggest impact on predictions?")
# st.pyplot(shap.plots.bar(st.session_state.shap_values, max_display=10))
# with st.expander("See explanation"):
#     st.write("""
#             We can order features by SHAP value, which indicates how much each feature (experience, degree, sales, etc.) 
#                     contributes to pushing the model's output (i.e., bonus) higher or lower.\n
#             In this plot, we use the mean absolute SHAP value $|ϕ_j|$ for each feature 
#                 over all the rows of the dataset as global measure of feature importance.
#             """)
        
# st.header("Beeswarm Plot")
# st.subheader("How does each feature affect predictions?")
# st.pyplot(shap.plots.beeswarm(st.session_state.shap_values, max_display=10))
# with st.expander("See explanation"):
#         st.write("""
#                 * Beeswarm plot used in SHAP library to interpret ML models.
#                 * The x-axis usually represents the SHAP value, 
#                     which indicates how much each feature (experience, degree, sales, etc.) 
#                     contributes to pushing the model's output (i.e., bonus) higher or lower.
#                 * Each dot on the plot represents a SHAP value of a single feature for a single prediction.
#                     Color indicates the value of the feature (red high, blow low).
#                 * Dots to the right of the center (usually 0) mean that the feature increases the prediction value. 
#                     Dots to the left mean the feature decreases the prediction value.\n
                
#                 In this plot, experience and degree have the highest spread of SHAP values,
#                 which makes them the most important features. 
#                 The higher the experience, the higher the SHAP values and therefore 
#                 the prediced bonus value. Or the higher the days late, 
#                 the lower the SHAP values and therefore the predicted bonus value.
#                 """)


    



