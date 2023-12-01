import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import openai  # Importing openai module
import os
import requests
from PIL import Image
import os

# Load the trained models
decision_tree_model = joblib.load('models/decision_tree_model.joblib')
linear_regression_model = joblib.load('models/linear_regression_model.joblib')
neural_network_model = joblib.load('models/neural_network_model.joblib')
random_forest_model = joblib.load('models/random_forest_model.joblib')

# Load the historical data
data = pd.read_csv('models/ethereum_historical_data.csv')

# Get the most recent price from the dataset
current_price = data['price'].iloc[-1]

# Function to get prediction intervals
def get_prediction_intervals(model, X, percentile=95):
    preds = np.stack([tree.predict(X) for tree in model.estimators_])
    lower = np.percentile(preds, (100 - percentile) / 2, axis=0)
    upper = np.percentile(preds, 100 - (100 - percentile) / 2, axis=0)
    return lower, upper

# Updated get_predictions to also provide confidence intervals for Random Forest
def get_predictions(features):
    dt_pred = decision_tree_model.predict(features)
    lr_pred = linear_regression_model.predict(features)
    nn_pred = neural_network_model.predict(features)
    rf_pred = random_forest_model.predict(features)
    
    # Calculate prediction intervals for Random Forest
    rf_lower, rf_upper = get_prediction_intervals(random_forest_model, features)
    
    return dt_pred, lr_pred, nn_pred, rf_pred, rf_lower, rf_upper

# Mock function for future features
def generate_future_features():
    return pd.DataFrame({
        '30_day_avg': [current_price] * 10,
        '10_day_avg': [current_price] * 10,
        'daily_return': [0.02] * 10
    })

future_features = generate_future_features()
dt_pred, lr_pred, nn_pred, rf_pred, rf_lower, rf_upper = get_predictions(future_features)

# Initialize OpenAI API key
openai.api_key = st.secrets['openai']['api_key']

# Function to generate text-based chat responses using OpenAI's GPT-3
def generate_text(prompt):
    headers = {
        "Authorization": f"Bearer {st.secrets['openai']['api_key']}",
        "Content-Type": "application/json"
    }

    endpoint = "https://api.openai.com/v1/chat/completions"
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        content = response.json()
        assistant_message = content['choices'][0]['message']['content']
        return assistant_message
    else:
        return f"Error {response.status_code}: {response.text}"

def main():
    st.set_page_config(page_title="Advanced TimeLock Wallet: ETH Prediction", layout="wide", initial_sidebar_state="collapsed")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.title('Advanced TimeLock Wallet: ETH Prediction Page')
        st.write("""
        Dive deep into Ethereum predictions. Explore various models, visualize historical data, and get insights tailored for the 
        Advanced TimeLock Wallet users.
        """)

        # Model Selection and Customization
        model = st.selectbox("Choose a Prediction Model", ["Linear Regression", "Decision Tree", "Neural Network", "Random Forest"])

        if model == "Linear Regression":
            prediction = lr_pred
        elif model == "Decision Tree":
            prediction = dt_pred
        elif model == "Neural Network":
            prediction = nn_pred
        else:
            prediction = rf_pred


        st.write(f"Today's Ethereum Price: ${current_price}")
        st.success(f"Predicted Ethereum Prices for the next 10 days using {model}: {prediction}")

# Interactive Visualization
        days = list(range(1, 11))
        prediction_df = pd.DataFrame({
            'Day': days,
            'Predicted_Price': rf_pred,   # Assuming Random Forest is the last in the tuple
            'Lower_Bound': rf_lower,
            'Upper_Bound': rf_upper
        })

        fig = px.line(prediction_df, x='Day', y='Predicted_Price',
                    line_dash_sequence=['solid'], labels={'value': 'Price (USD)'})
        fig.add_scatter(x=prediction_df['Day'], y=prediction_df['Lower_Bound'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)')
        fig.add_scatter(x=prediction_df['Day'], y=prediction_df['Upper_Bound'], fill='tonexty', fillcolor='rgba(0,100,80,0.2)')

        st.plotly_chart(fig)

    with col2:
        st.title("Educational Content")
        st.write("""
        **Linear Regression**: A basic predictive modeling technique that establishes a relationship between two variables.
        **Decision Tree**: A decision support tool that uses a tree-like model of decisions and their possible consequences.
        **Neural Network**: Computational systems inspired by the neural networks found in brains, used to estimate functions that depend on a large amount of unknown inputs.
        **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees during training and outputs the average prediction of the individual trees for regression problems.
        """)

        st.warning("""
        The cryptocurrency market is volatile. Ensure you make informed decisions and do not solely rely on 
        one prediction model. It's recommended to cross-check with multiple sources.
        """)

    # OpenAI chatbot section outside of the column context
    st.subheader("Ask our AI Assistant")
    st.write("Have questions about these models? Ask our AI assistant for more information.")

    # Store chatbot generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Clear chat history
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.button('Clear Chat History', on_click=clear_chat_history)

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a new text response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_text(prompt)
                    placeholder = st.empty()
                    placeholder.markdown(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

if __name__ == '__main__':
    main()