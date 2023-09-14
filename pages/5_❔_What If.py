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
st.set_page_config(page_title="What If", page_icon="❓")

st.set_option('deprecation.showPyplotGlobalUse', False)



st.header("Prediction")
selected_emp = st.selectbox("Employee",
                            st.session_state.data['employee_name'][:30],
                            label_visibility=st.session_state.visibility,
                            disabled=st.session_state.disabled,
                            key="wi_employee")
emp_idx = st.session_state.data['employee_name'][:30].tolist().index(selected_emp)
emp_pred = st.session_state.y_pred[emp_idx]
emp_pred_df = pd.DataFrame(data={"Bonus": [emp_pred]},
                                    index=["Predicted Bonus"])
st.dataframe(emp_pred_df, use_container_width=True)
st.header("Feature Input")
st.caption("Adjust feature values to see how they affect the predicted bonus")

col1, col2, col3 = st.columns(3)
with col1:
    wi_experience = st.number_input("Experience",
                    min_value=0,
                    max_value=42,
                    value=int(st.session_state.shap_values[emp_idx , "experience"].data),
                    step=1,
                    key="wi_experience",)
    wi_sales = st.number_input("Sales",
                                min_value= 0,
                                max_value= int(max(st.session_state.data['sales'])),
                                value=int(st.session_state.shap_values[emp_idx, "sales"].data),
                                step=1,
                                key="wi_sales",)
with col2:
    wi_degree = st.number_input("Degree",
                                min_value= min(st.session_state.data['degree']),
                                max_value= max(st.session_state.data['degree']),
                                value=int(st.session_state.shap_values[emp_idx, "degree"].data),
                                step=1,
                                key="wi_degree",)
    wi_days_late = st.number_input("Days Late",
                                min_value= 0,
                                max_value= 40,
                                value=int(st.session_state.shap_values[emp_idx, "days_late"].data),
                                step=1,
                                key="wi_days_late",)
with col3:
    wi_performance = st.number_input("Performance",
                                    min_value= min(st.session_state.data['performance']),
                                    max_value= max(st.session_state.data['performance']),
                                    value=st.session_state.shap_values[emp_idx, "performance"].data,
                                    step=0.01,
                                    key="wi_performance",)
    #if wi_generate_pred:
    wi_df = pd.DataFrame(data={"experience": [wi_experience],
                                    "degree": [wi_degree],
                                    "performance": [wi_performance],
                                    "sales": [wi_sales],
                                    "days_late": [wi_days_late]})
    wi_pred = st.session_state.model.predict(wi_df)
    st.metric(label="Predicted Bonus", 
                value=sum(wi_pred), 
                help=f"How much bonus {selected_emp} will get if the feature values are adjusted?")

st.subheader("Waterfall Plot")
st.caption(f"How has each feature contributed to {selected_emp}'s bonus after adjusting features?")
wi_shap_values = st.session_state.explainer(wi_df, check_additivity=False)
st.pyplot(shap.plots.waterfall(wi_shap_values[0], max_display=10))
    