import streamlit as st
import openai
import json

# Set your OpenAI API key here
openai.api_key = ""

# Streamlit page configuration
st.set_page_config(page_title='Pubrio JSON Extraction Tool', layout='wide')

def summarize_content(content, summary_length=1000):
    """
    Summarize the content to a specified length using OpenAI's summarization capabilities.

    Parameters:
    - content: The original content to summarize.
    - summary_length: The maximum number of tokens for the summary.

    Returns:
    - A summary of the original content.
    """
    # Summarization prompt
    summarize_prompt = f"Please summarize the following content into a concise summary of no more than {summary_length} tokens:\n\n{content}"

    # Call the OpenAI API to summarize the content
    summary_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=summarize_prompt,
        max_tokens=summary_length,
        n=1,
        stop=None,
        temperature=0.3,
    )

    # Extract the summarized content
    summarized_content = summary_response.choices[0].text.strip()
    return summarized_content

def extract_information(content):
    """
    Function to send a prompt to the OpenAI API and extract the required information.
    """
    # First, summarize the content to reduce its length due to limited tokens
    summarized_content = summarize_content(content)

    prompt = f"""
    Given the following article content, identify any mentioned companies and their domains, and summarize the main topic of the article. Then, present the information in a JSON format with two keys: "related_companies", which is an array of objects containing "company_name" and "company_domain", and "topic", which is a string describing the main topic or announcement of the article.

    Article content:
    {summarized_content}

    Please structure the information as follows:
    {{
      "related_companies": [
        {{
          "company_name": "Name of the first company",
          "company_domain": "Domain of the first company"
        }},
        {{
          "company_name": "Name of the second company",
          "company_domain": "Domain of the second company"
        }}
      ],
      "topic": "The main topic or announcement of the article"
    }}
    """

    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=150,
      n=1,
      stop=None,
      temperature=0.5,
    )
    
    return response.choices[0].text.strip()




def main():
    # Streamlit app main layout
    st.title('Pubrio JSON Extraction Tool')
    st.write('Enter the content of a website to extract related companies and the main topic.')

    # Text area for user input
    content = st.text_area("Website Content", height=300)
    
    if st.button('Extract Information'):
        with st.spinner('Extracting information...'):
            try:
                # Call the function to extract information
                result = extract_information(content)
                
                # Display the raw response for debugging
                st.text("Raw OpenAI API Response:")
                st.write(result)
                
                # Try to parse the raw response as JSON
                json_data = json.loads(result)
                st.json(json_data)
            except json.JSONDecodeError:
                st.error("Failed to decode the response into JSON. The response was not in the expected JSON format.")
                st.text("Raw text that failed to parse as JSON:")
                st.write(result)  # Show the raw result that failed to parse
            except openai.error.OpenAIError as e:
                st.error(f"An error occurred with the OpenAI API: {e}")

if __name__ == "__main__":
    main()
