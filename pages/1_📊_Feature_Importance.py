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

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Feature Importance", page_icon="📊")


st.header("Feature Descriptions")
st.caption("A brief description of each feature")
temp_df = pd.DataFrame(
[        
{"Feature": "experience", "Description": "Years of experience in the industry"},
{"Feature": "degree", "Description": "Level of education"},
{"Feature": "sales", "Description": "Sales performance"},
{"Feature": "performance", "Description": "Overall performance"},
{"Feature": "days_late", "Description": "Number of days late on projects"},
]
)
st.dataframe(temp_df, hide_index=True)

st.header("SHAP Summary")
st.subheader("Which features have the biggest impact on predictions?")
st.pyplot(shap.plots.bar(st.session_state.shap_values, max_display=10))
with st.expander("See explanation"):
    st.write("""
            We can order features by SHAP value, which indicates how much each feature (experience, degree, sales, etc.) 
                    contributes to pushing the model's output (i.e., bonus) higher or lower.\n
            In this plot, we use the mean absolute SHAP value $|ϕ_j|$ for each feature 
                over all the rows of the dataset as global measure of feature importance.
            """)
        
st.header("Beeswarm Plot")
st.subheader("How does each feature affect predictions?")
st.pyplot(shap.plots.beeswarm(st.session_state.shap_values, max_display=10))
with st.expander("See explanation"):
        st.write("""
                * Beeswarm plot used in SHAP library to interpret ML models.
                * The x-axis usually represents the SHAP value, 
                    which indicates how much each feature (experience, degree, sales, etc.) 
                    contributes to pushing the model's output (i.e., bonus) higher or lower.
                * Each dot on the plot represents a SHAP value of a single feature for a single prediction.
                    Color indicates the value of the feature (red high, blow low).
                * Dots to the right of the center (usually 0) mean that the feature increases the prediction value. 
                    Dots to the left mean the feature decreases the prediction value.\n
                
                In this plot, experience and degree have the highest spread of SHAP values,
                which makes them the most important features. 
                The higher the experience, the higher the SHAP values and therefore 
                the prediced bonus value. Or the higher the days late, 
                the lower the SHAP values and therefore the predicted bonus value.
                """)