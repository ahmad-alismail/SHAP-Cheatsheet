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
st.set_page_config(page_title="Individual Predictions", page_icon="👥")

st.set_option('deprecation.showPyplotGlobalUse', False)


residuals_diff = st.session_state.y - st.session_state.y_pred

st.header("Prediction")
selected_employee = st.selectbox("Employee",
                                    st.session_state.data['employee_name'][:30],
                                    label_visibility=st.session_state.visibility,
                                    disabled=st.session_state.disabled,
                                    key="ip_employee")
employee_idx = st.session_state.data['employee_name'][:30].tolist().index(selected_employee)
employee_pred = st.session_state.y_pred[employee_idx]
employee_actual = st.session_state.y[employee_idx]
employee_residual = residuals_diff[employee_idx]
base_bonus = round(st.session_state.shap_values[employee_idx].base_values,2)
experience_value = round(st.session_state.shap_values[employee_idx, "experience"].data,2)
experience_contrib = round(st.session_state.shap_values[employee_idx, "experience"].values,2)
individual_preds = pd.DataFrame(data={"Bonus": [employee_pred, employee_actual, employee_residual]},
                                    index=["Predicted", "Actual", "Residuals"])
st.dataframe(individual_preds, use_container_width=True)
st.subheader("Waterfall Plot")
st.caption(f"How has each feature contributed to {selected_employee}'s bonus?")
st.pyplot(shap.plots.waterfall(st.session_state.shap_values[employee_idx], max_display=10))
with st.expander("See explanation"):
    st.write(f"""
        Waterfall plot visualizes SHAP values as arrows that either increse or decrease 
                model prediction $f(x)$ compared to expected prediction $E[f(X)]$.\n
        Each feature pushes the model output from the **base value $E[f(x)]$** (the average model
                output over the training dataset we passed) to the model output $f(x)$.\n
        Features pushing the prediction higher are shown in red, 
                those pushing the prediction lower are in blue.\n
        In this example, we get from the base bonus {base_bonus} 
        to the predicted bonus {employee_pred}, where the experience of 
        {experience_value} pushed the prediction by {experience_contrib}    
    """)

st.subheader("Force Plot")
st.caption(f"How has each feature contributed to {selected_employee}'s bonus?")
st.pyplot(shap.plots.force(st.session_state.shap_values[employee_idx], matplotlib=True))
with st.expander("See explanation"):
        st.write("""
        Force plot shows the same information as the waterfall plot
        but in a different format.\n
                """
                )