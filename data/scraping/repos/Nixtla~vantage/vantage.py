# Import required libraries
import datetime
import os
from collections import defaultdict

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Set OpenAI API key from environment variable
openai.api_key = os.environ.get('OPENAI_TOKEN')


################################################
#  Define helper functions
################################################

def format_custom_date(dates_list):
    """Format a list of dates into a more readable format."""
    formatted_dates = []
    for date_str in dates_list:
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = date_obj.strftime("%A")
        day = date_obj.strftime("%d")
        day_suffix = "th" if 11 <= int(day) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(int(day) % 10, "th")
        month = date_obj.strftime("%B")
        year = date_obj.strftime("%Y")
        formatted_date = f"{day_of_week} {day}{day_suffix} of {month} {year}"
        formatted_dates.append(formatted_date)
    return formatted_dates


def explain_anomalies(anomaly_dates, service='Cloud services'):
    """Use OpenAI GPT-4 to generate explanations for anomalies in a given service."""
    response= openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in anomaly detection for time series. In particular you help people understand pontential explanations for anomalies in usage of cloud services. The user gives you dates on which they saw spikes and you analyze the dates and explain what could be special about those dates. For example, if they are holidays or close to holidays, quarted ends, month ends, weekends, etc. You always answer in a short parragprah and are concise. You always begin by saying: You saw a spike on your [service] usage on the following dates: and then you list the dates. You then explain what could be special about those dates related to that specifc service."},
            {"role": "user", "content": f"Here is the list of dates where I detected spikes in usage in {service}:  {format_custom_date(anomaly_dates)}"}
        ]
    )
    return response.choices[0].message.content

# Function to set the stage state to a specific value.
def set_state(i):
    st.session_state.stage = i

# Function to fetch data from a specific URL with headers.
# The response is cached for 1000 seconds to prevent repeated requests.
@st.cache_data(ttl=1000)
def fetch_data(url, headers):
    # Send a GET request to the specified URL.
    response = requests.get(url, headers=headers)
    
    try:
        # If the response indicates an error, raise an exception.
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        # If an HTTP error occurs, display a warning and stop the app.
        st.warning(f'HTTP error occurred: {err}. \n Please enter a valid request.')
        st.stop()
    
    # Return the JSON response.
    return response.json()

# Function to fetch reports from a specific URL with headers.
# The response is cached for 1000 seconds to prevent repeated requests.
@st.cache_data(ttl=1000)
def fetch_reports(url, headers):
    # This function works similarly to fetch_data().
    response = requests.get(url, headers=headers)
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        st.warning(f'HTTP error occurred: {err}. \n Please enter a valid request.')
        st.stop()
    
    return response.json()

# Function to create a figure with a specific title and axis labels.
def create_figure(title, xaxis_title, yaxis_title, yaxis_range=None):
    # Create a new Plotly Figure.
    fig = go.Figure()
    
    # Update the layout of the figure with the specified parameters.
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        autosize=False,
        width=800,
        height=500,
        yaxis=dict(range=yaxis_range) if yaxis_range else None,
    )
    
    # Return the figure.
    return fig

# Function to add a scatter trace to a figure.
def add_trace(fig, x, y, mode, name):
    # Add a scatter trace with the specified parameters.
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name))
    
    # Return the figure with the added trace.
    return fig

# Function to add a confidence interval to a figure.
def add_confidence_interval(fig, x, lo, hi, color='rgba(0,176,246,0.2)'):
    # Create a scatter trace for the confidence interval.
    fig.add_trace(go.Scatter(
        x=x + x[::-1],  # X coordinates for the filled area.
        y=hi + lo[::-1],  # Y coordinates for the filled area.
        fill='toself',  # The area under the trace is filled.
        fillcolor=color,  # The fill color.
        line_color='rgba(255,255,255,0)',  # The line color.
        showlegend=False,  # The trace is not added to the legend.
        name='Confidence Interval',
    ))
    
    # Return the figure with the added confidence interval.
    return fig
# Modify the add_confidence_interval function to mark points outside the confidence interval in red
def add_confidence_interval_anomalies(fig, historic_data, x, lower_bound, upper_bound):
    # Add the lower and upper bounds of the confidence interval as lines to the figure
    fig.add_trace(go.Scatter(x=x, y=lower_bound, fill=None, mode='lines', line_color='rgba(68, 68, 68, 0.2)', name='90% Confidence Interval'))
    fig.add_trace(go.Scatter(x=x, y=upper_bound, fill='tonexty', mode='lines', line_color='rgba(68, 68, 68, 0.2)', name='90% Confidence Interval'))

    # Get the y-values for the last set of data
    y_vals = list(historic_data["y"].values())[-len(upper_bound):]
    
    # Create a list of booleans that is True when the corresponding y value is above the upper bound
    above_confidence_interval = [y > upper for y, upper in zip(y_vals, upper_bound)]
    
    # Add to the figure points above the confidence interval marked in red
    fig.add_trace(go.Scatter(
        x=[x_val for x_val, above in zip(x[-len(upper_bound):], above_confidence_interval) if above],
        y=[y_val for y_val, above in zip(y_vals, above_confidence_interval) if above],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Above Confidence Interval'
    ))
    return fig

def create_exogenous_variable(series, horizon):
    """Create exogenous variable (binary indicator for start of each month)."""
    # Convert the input series to a pandas DataFrame
    df = pd.DataFrame(list(series["y"].items()), columns=['date', 'value'])
    
    # Convert the 'date' column to pandas datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Generate a list of dates for the entire period, including the horizon
    date_range = pd.date_range(start=df['date'].min(), periods=len(df) + horizon)
    
    # Create the exogenous variable dictionary with initial values as 0
    exogenous_variable = {date.strftime('%Y-%m-%d'): [0] for date in date_range}
    
    # Set the value to 1 for the initial date of each month in the exogenous variable
    for i in range(len(date_range)):
        if date_range[i].day == 1:
            exogenous_variable[date_range[i].strftime('%Y-%m-%d')][0] = 1
    
    return exogenous_variable

@st.cache_data(ttl=15)
def time_gpt(url, data, add_ex=True, token=os.environ.get('NIXTLA_TOKEN_PROD')):
    """Fetch time series forecasting results from Nixtla."""
    if add_ex:
        # If add_ex is True, create and add the exogenous variable to the data
        data["x"] = create_exogenous_variable(data, data["fh"])
    else:
        data["x"] = {}
    # Send a POST request to the specified URL.
    response = requests.post(url, json=data, headers={"authorization": f"Bearer {token}"})
    try:
        # If the response indicates an error, raise an exception.
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        # If an HTTP error occurs, display a warning and return None.
        st.warning(f'HTTP error occurred: {err}')
        return None
    
    # Return the JSON response.
    return response.json()

def get_anomalies(historic_data, insample_data):
    """Identify anomalies that exceed the 90% confidence interval."""
    y = list(historic_data["y"].values())[-len(insample_data['hi-90']):]
    y_keys = list(historic_data["y"].keys())[-len(insample_data['hi-90']):]
    anomaly_dates = [y_keys[i] for i, (y, upper) in enumerate(zip(y, insample_data['hi-90'])) if y > upper]
    return anomaly_dates

def transform_data(grouping, data_service):
    # Data transformation
    service_data = defaultdict(list)
    if grouping == 'provider':
        for cost in data_service["costs"]:
            date = pd.to_datetime(cost["accrued_at"])
            service_data[cost["provider"]].append((date, float(cost["amount"])))
    elif grouping == 'service':
        for cost in data_service["costs"]:
            date = pd.to_datetime(cost["accrued_at"])
            service_data[cost["service"]].append((date, float(cost["amount"])))
    elif grouping == 'account_id':
        for cost in data_service["costs"]:
            date = pd.to_datetime(cost["accrued_at"])
            service_data[cost["account_id"]].append((date, float(cost["amount"])))
    else:
        #Raise error because grouping is not supported
        st.error('Grouping is not supported. Please select a either service, provider or account_id')
    return service_data


################################################ Start of Streamlit app ################################################

st.set_page_config(page_title="Vantage+TimeGPT", page_icon="🚀", layout="centered", initial_sidebar_state="auto", menu_items=None)


# Check if 'stage' is in the session state. If not, initialize it to 0.
if 'stage' not in st.session_state:
    st.session_state.stage = 0

# Check if 'processed' key exists in the Streamlit session state, if not, initialize it to an empty dictionary.
if 'processed' not in st.session_state:
    st.session_state.processed = {}

# Set the title of the Streamlit app.
st.title('Forecasting Cloud Costs with Vantage and Nixtla')

# Write a welcoming message to the Streamlit app.
st.write('''
👋 Welcome to Vantage and Nixtla's forecasting app, your one-stop 🎯 solution for predicting ☁️ cloud costs with precision. Seamlessly integrated with Vantage's cloud cost transparency 💰 and Nixtla's advanced 📊 forecasting capabilities, this app takes the guesswork out of cloud budgeting. 🚀
''')

# Add a subheader to the Streamlit app.
st.subheader('Get your cloud costs with Vantage')

# Get the Vantage token from the user. The text_input function provides a text input box with the label 'Token:'.
# The second argument is the default value, which is 'vntg_tkn_c3f76e12ca64a4e9fadbd9037bc740cc3fde8b9d'.
vantage_token = st.text_input('Token:', 'vntg_tkn_c3f76e12ca64a4e9fadbd9037bc740cc3fde8b9d')

# If the user did not change the default Vantage token, show a warning and use an environment variable instead.
if vantage_token == 'vntg_tkn_c3f76e12ca64a4e9fadbd9037bc740cc3fde8b9d':
    st.warning('Using synthetic data. Please enter your Vantage token.')
    vantage_token = os.environ.get('VANTAGE_TOKEN')


################################################
# Get and Forecast cost report grouped by account_id
################################################

st.write("**See available reports:**")

# Create a button for fetching reports
if st.button('Get reports'):
    # Define the API endpoint and headers
    url = "https://api.vantage.sh/v1/reports"
    headers = {"accept": "application/json", "authorization": f"Bearer {vantage_token}"}
    
    # Show a spinner while fetching data
    with st.spinner('Fetching reports...'):
        # Call the previously defined function 'fetch_reports'
        data = fetch_reports(url, headers)

    # Extract the 'reports' list from the JSON response
    reports = data['reports']
    st.session_state.processed['reports'] = reports

    # Convert the 'reports' list into a DataFrame
    df = pd.DataFrame(reports)

    # Select only the 'id', 'title', and 'workspace' columns from the DataFrame
    df = df[['id', 'title', 'workspace']]

    # Display the DataFrame as a table in Streamlit
    st.table(df)

st.write("**Report ID to get cost details:**")

# User input for report ID
report_id = st.text_input('Enter Report ID:', '3637')
if st.button('Fetch historic data'):
    # Show spinner while fetching data
    with st.spinner('Fetching data from the API...'):
        url = f"https://api.vantage.sh/v1/reports/{report_id}/costs?grouping=account_id&?start_date=2023-03-01"
        headers = {"accept": "application/json", "authorization": f"Bearer {vantage_token}"}
        data = fetch_data(url, headers)

        # Transform the data into a dictionary for future forecasting
        historic_data = {"y": {}, "fh": 30, "level": [90], "finetune_steps": 2}
        for cost in data["costs"]:
            historic_data["y"][cost["accrued_at"]] = float(cost["amount"])
        
        st.session_state['historic_data'] = historic_data
        st.success('Costs fetched successfully!')
        st.session_state.processed['historic_data'] = historic_data

    if not st.session_state.processed['historic_data']:
        st.warning('Please fetch data first.')

st.write("**Forecast costs and Detect anomalies:**")
if st.button('Forecast costs and Detect anomalies'):
    try :
        assert st.session_state.processed['historic_data']
    except KeyError:
        st.warning('Please fetch data first.')
        st.stop()
    # Request forecast from time GPT
    with st.spinner('🔮 Forecasting... 💾 Hang tight! 🚀'):
        post_url = os.environ.get('LTM1_PROD')
        ### HERE IS WHERE THE MAGIC HAPPENS ###
        #st.header('PAYLOAD')
        #st.write(st.session_state.processed['historic_data'])
        new_data = time_gpt(post_url, st.session_state.processed['historic_data'], add_ex=True)
        #st.header('RESPONSE')
        #st.write(new_data)
        if new_data:
            st.success('✅ Forecasting completed successfully!')
            new_data = new_data['data']
        else:
            st.stop()

    # Visualization
    with st.spinner('👩‍💻 Plotting'):
        fig = create_figure('Current and Forecasted Cloud Costs', 'Date', 'Spend in USD')
        fig = add_trace(fig, list(st.session_state.processed['historic_data']["y"].keys()), list(st.session_state.processed['historic_data']["y"].values()), 'lines', 'Original Data')
        fig = add_trace(fig, new_data['timestamp'], new_data['value'], 'lines', 'Forecasted Data')
        fig = add_confidence_interval(fig, new_data['timestamp'], new_data['lo-90'], new_data['hi-90'])
        st.plotly_chart(fig)
        


    ################################################
    # Detect Anomalies for the selected report grouped by account_id
    ################################################

    # In-sample predictions
    st.header('Anomaly detection with Vantage and Nixtla')
    st.write( '''
    This app leverages the power of Vantage's robust data analytics platform 💼 and Nixtla's cutting-edge forecasting techniques 📈 to identify outliers in your data in real-time. 🔍  You can view available reports 📋, input specific report IDs 🔢 for more detailed insights, and even fetch cost details 💰 on demand. So go ahead, explore your data 🔎, and let's unveil the hidden anomalies together! 😎
    ''')

    with st.spinner('🔎 Detecting anomalies...'):
        # Fetching in-sample predictions
        insample_post_url = os.environ.get('INSAMPLE_LTM_URL_PROD')
        insample_data = time_gpt(insample_post_url, st.session_state.processed['historic_data'], add_ex=False, token=os.environ.get('NIXTLA_TOKEN_PROD'))
        insample_data = insample_data['data']

        # Creating the plot for in-sample predictions
        fig_insample = create_figure('Current and In-sample Predicted Cloud Costs', 'Date', 'Spend in USD')
        fig_insample = add_trace(fig_insample, list(st.session_state.processed['historic_data']["y"].keys()), list(st.session_state.processed['historic_data']["y"].values()), 'lines', 'Original Data')
        fig_insample = add_trace(fig_insample, insample_data['timestamp'], insample_data['value'], 'lines', 'In-sample Predictions')
        fig_insample = add_confidence_interval_anomalies(fig_insample, st.session_state.processed['historic_data'], insample_data['timestamp'], insample_data['lo-90'], insample_data['hi-90'])
        st.plotly_chart(fig_insample)

        # Detecting anomalies based on the confidence interval of in-sample predictions
        y = list(st.session_state.processed['historic_data']["y"].values())[-len(insample_data['hi-90']):]
        y_keys = list(st.session_state.processed['historic_data']["y"].keys())[-len(insample_data['hi-90']):]
        anomalies_list = [y > upper for y, upper in zip(y, insample_data['hi-90'])]
        anomaly_dates = [y_keys[i] for i, (y, upper) in enumerate(zip(y, insample_data['hi-90'])) if y > upper]

    # Explaining detected anomalies
    with st.spinner('🔎 Explaining anomalies with Open AI... \n 🤖 We use GPT4, so this might take some minutes...'):

        st.write(explain_anomalies(anomaly_dates))
        st.balloons()


################################################
#  Get and Forecast cost report grouped by service or provider
################################################


# Display a header in the application.
st.header('Select a specific grouping criteria to forecast its future costs')

# Take inputs from the user for the start date, grouping criteria, and report ID.
start_date = st.text_input('Start date', value='2023-03-01')
grouping = st.text_input('Grouping', value='provider')
report_id = st.text_input('Report ID', value= '')

# If the report ID is not provided, display a warning.
if report_id == '':
    st.warning('Please enter a valid report ID')
else:
    # Display a spinner to indicate that the data is being fetched and the plot is being created.
    with st.spinner('Fetching data and creating the plot...'):
        # Fetch the data for the selected service.
        url_service = f"https://api.vantage.sh/v1/reports/{report_id}/costs?grouping={grouping}&?start_date={start_date}"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {vantage_token}"
        }
        historic_data_grouped = fetch_data(url_service, headers)
        st.session_state.processed['historic_data_grouped'] = historic_data_grouped

    # Display a spinner to indicate that the data is being transformed.
    with st.spinner('Transforming the data for the selected service...'):
        # Transform the data based on the selected grouping criteria.
        service_data = transform_data(grouping, historic_data_grouped)

    # Display a spinner to indicate that the forecast is being created.
    with st.spinner('🔮 Forecasting... 💾 Hang tight! 🚀'):
        # Initialize the selected service if it has not been selected before.
        if 'st.session_state.selected_service' not in st.session_state:
            st.session_state.selected_service = 0  # default to the first service

        # Allow the user to select a service.
        st.session_state.selected_service = st.selectbox('Select a service or provider:', list(service_data.keys()), st.session_state.selected_service)
        selected_dates, selected_values = zip(*service_data[st.session_state.selected_service])

        # Create a figure for the selected service's data.
        fig_service = create_figure(f'Costs and Forecast for {st.session_state.selected_service}', 'Date', 'Spend in USD', [0, max(selected_values)+10])
        fig_service = add_trace(fig_service, selected_dates, selected_values, 'lines', st.session_state.selected_service)

        # Prepare the data for the POST request.
        historic_data_grouped = {"y": {date.strftime('%Y-%m-%d'): value for date, value in zip(selected_dates, selected_values)}, "fh": 30, "level": [90], 'finetune_steps': 2}
        post_url = os.environ.get('LTM1_PROD_URL')
        new_data_grouped = time_gpt(post_url, historic_data_grouped)
        new_data_grouped = new_data_grouped['data']

        # Extract the forecast and confidence interval data.
        new_dates_service = [pd.to_datetime(date) for date in new_data_grouped['timestamp']]
        new_values_service = new_data_grouped['value']
        new_lo_service = new_data_grouped['lo-90'] if 'lo-90' in new_data_grouped else [0]*len(new_values_service)
        new_hi_service = new_data_grouped['hi-90'] if 'hi-90' in new_data_grouped else [0]*len(new_values_service)

        # Add the forecast and confidence interval data to the figure.
        fig_service = add_trace(fig_service, new_dates_service, new_values_service, 'lines', 'Forecasted Data')
        fig_service = add_confidence_interval(fig_service, new_dates_service, new_lo_service, new_hi_service)
        
        # Display the figure in the application.
        st.plotly_chart(fig_service)

    historic_data = st.session_state.processed['historic_data']

    st.header(f'Anomaly detections for {st.session_state.selected_service}')
    with st.spinner(f'Analyzing {st.session_state.selected_service} and detecting anomalies'):
        # Making in-sample predictions for the selected service and creating the plot logic...
        insample_post_url = os.environ.get('INSAMPLE_LTM_URL_PROD')
        insample_data_service = time_gpt(insample_post_url, historic_data_grouped, add_ex=False)
        insample_data_service = insample_data_service['data']

        # Create the figure for in-sample predictions
        fig_insample_service = create_figure(f'In-sample Predictions and Actual Costs for {st.session_state.selected_service}', 'Date', 'Spend in USD', [0, max(selected_values)+10])
        fig_insample_service = add_trace(fig_insample_service, selected_dates, selected_values, 'lines', f'Original Data ({st.session_state.selected_service})')
        fig_insample_service = add_trace(fig_insample_service, insample_data_service['timestamp'], insample_data_service['value'], 'lines', 'In-sample Predictions')

        # Add confidence interval if available in the data
        #if 'lo-90' in insample_data_service and 'hi-90' in insample_data_service:
        fig_insample_service = add_confidence_interval_anomalies(fig_insample_service, historic_data_grouped, insample_data_service['timestamp'], insample_data_service['lo-90'], insample_data_service['hi-90'])
        st.plotly_chart(fig_insample_service)
    with st.spinner('🔎 Explaining anomalies...'):
        # Get anomalies
        anomaly_services_dates= get_anomalies(historic_data_grouped, insample_data_service)
        st.write(explain_anomalies(anomaly_services_dates, service=st.session_state.selected_service))
        st.snow()
