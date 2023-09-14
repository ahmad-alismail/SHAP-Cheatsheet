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
st.set_page_config(page_title="Feature Dependence", page_icon="🔍")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("Dependence Scatter Plot")
st.subheader("Relationship between sigle feature value and SHAP value")
col1, col2 = st.columns(2)
with col1:
    basis_feature = st.selectbox("Basis Feature",
                                st.session_state.X_features,
                                label_visibility=st.session_state.visibility,
                                disabled=st.session_state.disabled,
                                key="dsp_base_feature")
with col2:
    interacted_feature = st.selectbox("Interacted Feature",
                                ["Most Interacted Feature"]+st.session_state.X_features,
                                label_visibility=st.session_state.visibility,
                                disabled=st.session_state.disabled,
                                key="dsp_inter_feature",
                                )
st.subheader(f"How does the predicted bonus change when the {basis_feature} changes?")
if interacted_feature == "Most Interacted Feature":
    st.pyplot(shap.plots.scatter(st.session_state.shap_values[:, basis_feature], color=st.session_state.shap_values))
else:
    st.pyplot(shap.plots.scatter(st.session_state.shap_values[:, basis_feature], color=st.session_state.shap_values[:, interacted_feature]))

with st.expander("See explanation"):
        st.write(f"""
        * Dependence scatter plot shows how SHAP values $ϕ_(i,j)$ (impact on model output) 
        change with increasing value of feature $j$ (i.e., {basis_feature}) across the whole dataset.\n
        * 1 dot per data instance.
        * Color based on the feature $k$ that interacts most with feature $j$. 
            You can change the interacted feature in the sidebar.\n
            
        
        In this example, the higher the **{basis_feature}**, the higher the SHAP values and therefore
        the predicted bonus value.\n
        > **Note**: The feature experience interacts the most with degree. That is
        an employee with degree and high experience will have higher bonus than an employee without
        degree and same level of experience.\n
        Another Example: the higher the days late, the lower the SHAP values and therefore
        the predicted bonus value.

    """)