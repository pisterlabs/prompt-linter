import os
import gradio
import pandas as pd
import openai

openai.api_key = os.getenv('OPENAI_KEY')

# Custom CSS
custom_css = f"""
        .gradio-header {{
            color: white;
        }}
        .gradio-description {{
            color: white;
        }}
        gradio-app {{
            background-image: url('https://raw.githubusercontent.com/kevinbtalbert/Electric_and_Utilities_System_Demo/main/CML-Assets/app_assets/cldr_bg.jpg') !important;
            background-size: cover  !important;
            background-position: center center  !important;
            background-repeat: no-repeat  !important;
            background-attachment: fixed  !important;
        }}
        #custom-logo {{
            text-align: center;
        }}
        .dark {{
            background-image: url('https://raw.githubusercontent.com/kevinbtalbert/Electric_and_Utilities_System_Demo/main/CML-Assets/app_assets/cldr_bg.jpg') !important;
            background-size: cover  !important;
            background-position: center center  !important;
            background-repeat: no-repeat  !important;
            background-attachment: fixed  !important;
        }}
        .gr-interface {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .gradio-header {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .gradio-input-box, .gradio-output-box {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        h1 {{
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: large; !important;
        }}
"""
def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses,
                            title="Electric & Utilities Company AI-Powered Assistant",
                            description="This AI-powered assistant is designed to help you understand outages in your area as well as be a source for questions about your utility company. For outages, served, and affected, simply enter the area name. You can ask complete questions for the chatbot.",
                            inputs=[gradio.Radio(['outages', 'customers-served', 'customers-affected', 'chatbot'], label="Select Use Case", value="outages"), gradio.Textbox(label="Area/Question", placeholder="")],
                            outputs=[gradio.Textbox(label="Response")],
                            allow_flagging="never",
                            css=custom_css)

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(option, question):
    engine = "gpt-3.5-turbo"

    if question is "" or question is None:
        return "No question and/or engine selected."
    
    if option == "outages":
        res = get_outages_by_area(question)
        context_chunk="You are a chatbot responding for an electric and utilities company."
        question = "Explain to me that the area " + question + " has " + str(res) + " outages."

    if option == "customers-served":
        res = get_customers_served_by_area(question)
        context_chunk="You are a chatbot responding for an electric and utilities company"
        question = "Explain to me that the area " + question + " has " + str(res) + " served customers."

    if option == "customers-affected":
        res = get_customers_affected_by_area(question)
        context_chunk = "You are a chatbot responding for an electric and utilities company. "
        question = "Explain to me that the area " + question + " has " + str(res) + " affected customers by outages."

    if option == "chatbot":
        context_chunk="You are a chatbot responding to a question for an electric and utilities company. If this question is not about that domain, say you cannot answer it: "

    # Perform text generation with LLM model
    response = get_llm_response(question, context_chunk, engine)

    return response

def get_outages_by_area(area_name):
    try:
        # Read the CSV file
        data = pd.read_csv('/home/cdsw/CML-Assets/data/utility_outage_data.csv')

        # Convert the 'Area Name' column to uppercase for case-insensitive comparison
        data['Area Name'] = data['Area Name'].str.upper()

        # Find the row with the matching area name
        area_data = data[data['Area Name'] == area_name.upper()]

        # Check if the area is found
        if not area_data.empty:
            # Return the number of outages
            return area_data.iloc[0]['Number of Outages']
        else:
            return "Area name not found."

    except FileNotFoundError:
        return "Outage data not found."
    
def get_customers_served_by_area(area_name):
    try:
        # Read the CSV file
        data = pd.read_csv('/home/cdsw/CML-Assets/data/utility_outage_data.csv')

        # Convert the 'Area Name' column to uppercase for case-insensitive comparison
        data['Area Name'] = data['Area Name'].str.upper()

        # Find the row with the matching area name
        area_data = data[data['Area Name'] == area_name.upper()]

        # Check if the area is found
        if not area_data.empty:
            # Return the number of customers served
            return area_data.iloc[0]['Customers Served']
        else:
            return "Area name not found."

    except FileNotFoundError:
        return "Outage data not found."
    
def get_customers_affected_by_area(area_name):
    try:
        # Read the CSV file
        data = pd.read_csv('/home/cdsw/CML-Assets/data/utility_outage_data.csv')

        # Convert the 'Area Name' column to uppercase for case-insensitive comparison
        data['Area Name'] = data['Area Name'].str.upper()

        # Find the row with the matching area name
        area_data = data[data['Area Name'] == area_name.upper()]

        # Check if the area is found
        if not area_data.empty:
            # Return the number of customers affected
            return area_data.iloc[0]['Approximate Customers Affected']
        else:
            return "Area name not found."

    except FileNotFoundError:
        return "Outage data not found."
    


# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llm_response(question, context, engine):
    
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": str(context)},
            {"role": "user", "content": str(question)}
            ]
    )
    
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    main()
