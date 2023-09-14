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
st.set_page_config(page_title="Regression Stats", page_icon="📈")

st.set_option('deprecation.showPyplotGlobalUse', False)


st.header("Model Summary")
st.subheader("Quantitative metrics for model performance")
st.caption("Metrics that measure how well the model is able to predict the bonus amount")
mae = round(mean_absolute_error(st.session_state.y, st.session_state.y_pred),2)
mse = round(mean_squared_error(st.session_state.y, st.session_state.y_pred), 2)
rmse = round(np.sqrt(mse),2)
r2 = round(r2_score(st.session_state.y, st.session_state.y_pred),2 )

col1, col2 = st.columns(2)

col1.metric(label="MAE", value=mae, help=f"On average, the model's predictions are off by {mae}$")
col1.metric(label="RMSE", value=rmse, help="How close the model's predictions are to the actual values, balancing the focus on larger errors while retaining original unit interpretability. The lower the better.")
col2.metric(label="MSE", value=mse, help="How close the model's predictions are to the actual values, amplifying larger mistakes, which makes it sensitive to outliers but harder to interpret.")
col2.metric(label="R2", value=r2, help="Measures how much better the model is compared to predicting the average.")
####################################################################     
st.header("Residuals")
st.subheader("How much the model 'missed' for each prediction?")
residuals_diff = st.session_state.y - st.session_state.y_pred
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=st.session_state.y_pred, 
                        y=residuals_diff,
                        marker=dict(color='#337CCF'),
                        mode='markers', 
                        name='Residuals',
                        text=st.session_state.data['employee_name']))
fig1.add_shape(
type='line',
line=dict(color='#FFC436', width=3),
x0=min(st.session_state.y_pred),
x1=max(st.session_state.y_pred),
y0=0,
y1=0
)
fig1.update_layout(
    title='Residual Plot',
    xaxis_title='Predicted Value',
    yaxis_title='Residual',
    template='plotly_white'
)
st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
with st.expander("See explanation"):
        st.write("""
        Residuals are the differences between what a model predicts and what actually happened. 
        \nIn this plot you can check if the model works better or worse for different bonus levels. 
        If the points are randomly scattered around a horizontal line (usually the zero line), 
        that suggests your model is fairly **accurate**. 
        On the other hand, if you see patterns, like a curve or a slope, 
        it suggests your model is **missing something** and could be improved.
    """)
#####################################################################    
st.header("Predicted vs. Actual Values")
st.subheader("How close is the predicted to the actual values?")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=st.session_state.y_pred, 
                            y=st.session_state.y, 
                            mode='markers', 
                            name='Predicted vs Actual', 
                            marker=dict(color='#337CCF'),
                            text=st.session_state.data['employee_name']))
fig2.add_shape(
    type='line',
    line=dict(color='#FFC436', width=3),
    x0=min(min(st.session_state.y_pred), min(st.session_state.y)),
    x1=max(max(st.session_state.y_pred), max(st.session_state.y)),
    y0=min(min(st.session_state.y_pred), min(st.session_state.y)),
    y1=max(max(st.session_state.y_pred), max(st.session_state.y))
)
fig2.update_layout(
    title='Predicted vs Actual Bonus',
    xaxis_title='Predicted Bonus',
    yaxis_title='Actual Bonus',
    template='plotly_white'
)
st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
with st.expander("See explanation"):
        st.write("""
        Plot shows the actual bonus amounts and the predicted ones on the same plot.\n
        In this plot you can check if the model works better or worse for different bonus levels.\n
        A perfect model would have all the points lying on the diagonal line 
        (i.e., predicted matches actual).     
        The further away points are from the diagonal line,
        the worse the model is at predicting those bonus values.
    """)
#################################################################################
st.header("Residuals vs. Features")
st.subheader("How feature values are correlated with residuals, predicted, actual bonus amounts?")
X_numpy = st.session_state.X.to_numpy()
col1, col2 = st.columns(2)
with col1:
    selected_feature = st.selectbox("Feature",
                                    st.session_state.X_features,
                                    label_visibility=st.session_state.visibility,
                                    disabled=st.session_state.disabled,
                                    key="rfp_feature")
feature_idx = st.session_state.X_features.index(selected_feature)
feature_values = X_numpy[:, feature_idx]
with col2:
        selected_yaxis = st.selectbox("Display",
                                ["Residuals (y - preds)", "Actual Bonus", "Predicted Bonus"],
                                label_visibility=st.session_state.visibility,
                                disabled=st.session_state.disabled,
                                key="rfp_yaxis")
if selected_yaxis == "Residuals (y - preds)":
    yaxis = residuals_diff
elif selected_yaxis == "Actual Bonus":
    yaxis = y
else:
    yaxis = st.session_state.y_pred

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=feature_values, 
                            y=yaxis, 
                            mode='markers', 
                            marker=dict(color='#337CCF'),
                            text=st.session_state.data['employee_name']))
if selected_yaxis == "Residuals (y - preds)":   
    fig3.add_shape(
        type='line',
        line=dict(color='#FFC436', width=3),
        x0=min(feature_values),
        x1=max(feature_values),
        y0=0,
        y1=0
        )
fig3.update_layout(
    title=f'{selected_yaxis} vs {selected_feature}',
    xaxis_title=selected_feature, 
    yaxis_title=f'{selected_yaxis}',
    template='plotly_white'
    )
st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

with st.expander("See explanation"):
        st.write("""
        This plot shows feature values plotted against either residuals 
        (difference between actual bonus and predicted bonus),
        actual bonus or predicted bonus.\n
        So you can inspect whether the model is more wrong for 
        particular range of features that others.\n
        In addition, how the feature values affect actual and predicted bonus values\n
        
    """)