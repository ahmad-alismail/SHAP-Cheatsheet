import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
# shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state variables
if 'data' not in st.session_state:
	st.session_state.data = pd.read_csv("./data/interaction_dataset_with_names.csv",sep='\t')
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title="Home", page_icon="🦜️🔗")

st.header("Regression Explainer")

tab1, tab2, tab3, \
	tab4, tab5, tab6 = st.tabs(["Feature Importances", 
							"Regression Stats",
							"Individual Predictions", 
							"Feature Dependence",
                            "What if?",
							"Feature Interactions"])



y = st.session_state.data['bonus']
X = st.session_state.data.drop(['bonus','employee_name'], axis=1)
X_features = list(X.columns)
model = RandomForestRegressor(n_estimators=100) 
model.fit(X, y)
y_pred = model.predict(X)
#Get SHAP values
explainer = shap.Explainer(model,X[:10])
shap_values = explainer(X, check_additivity=False)

with tab1:    
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
    st.pyplot(shap.plots.bar(shap_values, max_display=10))
    with st.expander("See explanation"):
        st.write("""
                We can order features by SHAP value, which indicates how much each feature (experience, degree, sales, etc.) 
                        contributes to pushing the model's output (i.e., bonus) higher or lower.\n
                In this plot, we use the mean absolute SHAP value $|ϕ_j|$ for each feature 
                    over all the rows of the dataset as global measure of feature importance.
                """)
         
    st.header("Beeswarm Plot")
    st.subheader("How does each feature affect predictions?")
    st.pyplot(shap.plots.beeswarm(shap_values, max_display=10))
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

with tab2:
    st.header("Model Summary")
    st.subheader("Quantitative metrics for model performance")
    st.caption("Metrics that measure how well the model is able to predict the bonus amount")
    mae = round(mean_absolute_error(y, y_pred),2)
    mse = round(mean_squared_error(y, y_pred), 2)
    rmse = round(np.sqrt(mse),2)
    r2 = round(r2_score(y, y_pred),2 )

    col1, col2 = st.columns(2)

    col1.metric(label="MAE", value=mae, help=f"On average, the model's predictions are off by {mae}$")
    col1.metric(label="RMSE", value=rmse, help="How close the model's predictions are to the actual values, balancing the focus on larger errors while retaining original unit interpretability. The lower the better.")
    col2.metric(label="MSE", value=mse, help="How close the model's predictions are to the actual values, amplifying larger mistakes, which makes it sensitive to outliers but harder to interpret.")
    col2.metric(label="R2", value=r2, help="Measures how much better the model is compared to predicting the average.")
    ####################################################################     
    st.header("Residuals")
    st.subheader("How much the model 'missed' for each prediction?")
    residuals_diff = y - y_pred
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=y_pred, 
                         y=residuals_diff,
                         marker=dict(color='#337CCF'),
                         mode='markers', 
                         name='Residuals',
                         text=st.session_state.data['employee_name']))
    fig1.add_shape(
    type='line',
    line=dict(color='#FFC436', width=3),
    x0=min(y_pred),
    x1=max(y_pred),
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
    fig2.add_trace(go.Scatter(x=y_pred, 
                              y=y, 
                              mode='markers', 
                              name='Predicted vs Actual', 
                              marker=dict(color='#337CCF'),
                              text=st.session_state.data['employee_name']))
    fig2.add_shape(
        type='line',
        line=dict(color='#FFC436', width=3),
        x0=min(min(y_pred), min(y)),
        x1=max(max(y_pred), max(y)),
        y0=min(min(y_pred), min(y)),
        y1=max(max(y_pred), max(y))
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
    X_numpy = X.to_numpy()
    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox("Feature",
                                        X_features,
                                        label_visibility=st.session_state.visibility,
                                        disabled=st.session_state.disabled,
                                        key="rfp_feature")
    feature_idx = X_features.index(selected_feature)
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
        yaxis = y_pred

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
         
with tab3:
    st.header("Prediction")
    selected_employee = st.selectbox("Employee",
                                        st.session_state.data['employee_name'][:30],
                                        label_visibility=st.session_state.visibility,
                                        disabled=st.session_state.disabled,
                                        key="ip_employee")
    employee_idx = st.session_state.data['employee_name'][:30].tolist().index(selected_employee)
    employee_pred = y_pred[employee_idx]
    employee_actual = y[employee_idx]
    employee_residual = residuals_diff[employee_idx]
    base_bonus = round(shap_values[employee_idx].base_values,2)
    experience_value = round(shap_values[employee_idx, "experience"].data,2)
    experience_contrib = round(shap_values[employee_idx, "experience"].values,2)
    individual_preds = pd.DataFrame(data={"Bonus": [employee_pred, employee_actual, employee_residual]},
                                     index=["Predicted", "Actual", "Residuals"])
    st.dataframe(individual_preds, use_container_width=True)
    st.subheader("Waterfall Plot")
    st.caption(f"How has each feature contributed to {selected_employee}'s bonus?")
    st.pyplot(shap.plots.waterfall(shap_values[employee_idx], max_display=10))
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
    st.pyplot(shap.plots.force(shap_values[employee_idx], matplotlib=True))
    with st.expander("See explanation"):
         st.write("""
            Force plot shows the same information as the waterfall plot
            but in a different format.\n
                  """
                    )
         
with tab4:
    st.header("Dependence Scatter Plot")
    st.subheader("Relationship between sigle feature value and SHAP value")
    col1, col2 = st.columns(2)
    with col1:
        basis_feature = st.selectbox("Basis Feature",
                                    X_features,
                                    label_visibility=st.session_state.visibility,
                                    disabled=st.session_state.disabled,
                                    key="dsp_base_feature")
    with col2:
        interacted_feature = st.selectbox("Interacted Feature",
                                    ["Most Interacted Feature"]+X_features,
                                    label_visibility=st.session_state.visibility,
                                    disabled=st.session_state.disabled,
                                    key="dsp_inter_feature",
                                    )
    st.subheader(f"How does the predicted bonus change when the {basis_feature} changes?")
    if interacted_feature == "Most Interacted Feature":
        st.pyplot(shap.plots.scatter(shap_values[:, basis_feature], color=shap_values))
    else:
        st.pyplot(shap.plots.scatter(shap_values[:, basis_feature], color=shap_values[:, interacted_feature]))

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
     
     