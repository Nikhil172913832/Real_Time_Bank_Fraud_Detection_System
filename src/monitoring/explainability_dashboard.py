"""
Model Explainability Dashboard
==============================

Streamlit dashboard for visualizing SHAP values and model explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Fraud Detection Explainability",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Fraud Detection Model Explainability Dashboard")
st.markdown("---")


@st.cache_resource
def load_model():
    """Load trained model and feature columns."""
    model = joblib.load("models/xgb_final.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns


@st.cache_data
def load_sample_data():
    """Load sample data for analysis."""
    try:
        df = pd.read_csv("data.csv").head(1000)
        return df
    except:
        return None


def calculate_shap_values(model, X):
    """Calculate SHAP values for given data."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Fraud class
    
    return explainer, shap_values


def main():
    # Load model
    try:
        model, feature_columns = load_model()
        st.sidebar.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Global Explainability", "Single Prediction", "Feature Importance"]
    )
    
    # Load data
    df = load_sample_data()
    
    if df is None:
        st.warning("No sample data available. Please upload data.")
        return
    
    # Feature engineering
    from src.features.engineering import FeatureEngineer
    engineer = FeatureEngineer(validate_schema=False)
    features = engineer.transform(df)
    X = engineer.prepare_for_model(features, encode_categoricals=True)
    
    # Align columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    
    # Calculate SHAP values
    with st.spinner("Calculating SHAP values..."):
        explainer, shap_values = calculate_shap_values(model, X)
    
    # Analysis
    if analysis_type == "Global Explainability":
        st.header("Global Feature Importance")
        
        # Summary plot
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig)
        
        # Feature importance table
        st.subheader("Top 20 Features")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False).head(20)
        
        st.dataframe(feature_importance, use_container_width=True)
        
    elif analysis_type == "Single Prediction":
        st.header("Single Transaction Explanation")
        
        # Select transaction
        transaction_idx = st.slider(
            "Select Transaction",
            0, len(X) - 1, 0
        )
        
        # Get prediction
        fraud_prob = model.predict_proba(X.iloc[[transaction_idx]])[:, 1][0]
        prediction = "🚨 FRAUD" if fraud_prob > 0.5 else "✅ LEGITIMATE"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        
        # SHAP waterfall plot
        st.subheader("Feature Contributions")
        
        # Get SHAP values for this transaction
        shap_dict = dict(zip(feature_columns, shap_values[transaction_idx]))
        shap_df = pd.DataFrame({
            'feature': list(shap_dict.keys()),
            'shap_value': list(shap_dict.values())
        }).sort_values('shap_value', key=abs, ascending=False).head(15)
        
        # Create bar chart
        fig = px.bar(
            shap_df,
            x='shap_value',
            y='feature',
            orientation='h',
            title="Top 15 Feature Contributions",
            color='shap_value',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature values
        st.subheader("Feature Values")
        feature_values = X.iloc[transaction_idx].to_dict()
        top_features = shap_df['feature'].tolist()
        
        feature_table = pd.DataFrame({
            'Feature': top_features,
            'Value': [feature_values.get(f, 'N/A') for f in top_features],
            'SHAP Value': [shap_dict[f] for f in top_features]
        })
        st.dataframe(feature_table, use_container_width=True)
        
    else:  # Feature Importance
        st.header("Feature Importance Analysis")
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Top 20 features
        top_20 = feature_importance.head(20)
        
        fig = px.bar(
            top_20,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Most Important Features",
            labels={'importance': 'Mean |SHAP Value|'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of SHAP values
        st.subheader("SHAP Value Distributions")
        
        selected_feature = st.selectbox(
            "Select Feature",
            top_20['feature'].tolist()
        )
        
        feature_idx = feature_columns.index(selected_feature)
        feature_shap = shap_values[:, feature_idx]
        
        fig = px.histogram(
            x=feature_shap,
            nbins=50,
            title=f"Distribution of SHAP Values for {selected_feature}",
            labels={'x': 'SHAP Value'}
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
