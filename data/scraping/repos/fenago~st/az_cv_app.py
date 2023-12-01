import streamlit as st
import openai
import json
from PIL import Image
import requests
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Initialize Azure OpenAI
openai.api_key = st.secrets["api_key"]
openai.api_base = "https://ttai2335.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
deployment_name = 'gpt-35-turbo'

model_engine = 'text-davinci-003'
deployment_name = 'gpt-35-turbo'
openai.api_key = st.secrets["api_key"]
az_key = st.secrets["az_key"]
# Initialize Computer Vision Client
computervision_client = ComputerVisionClient('https://cvdrlee.cognitiveservices.azure.com/', CognitiveServicesCredentials(az_key))

def analyze_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    
    # Reset the pointer of BytesIO object to the start
    img_byte_arr.seek(0)
    
    # Use BytesIO object directly as it's a type of stream
    analysis = computervision_client.analyze_image_in_stream(img_byte_arr, visual_features=["Categories", "Tags", "Description", "Color", "ImageType"])
    
    return analysis


def display_analysis(analysis):
    st.write("### Categories:")
    for category in analysis.categories:
        st.write(f"  {category.name} ({category.score*100:.2f}%)")
    st.write("### Tags:")
    for tag in analysis.tags:
        st.write(f"  {tag.name} ({tag.confidence*100:.2f}%)")
    st.write("### Description:")
    st.write(f"  {analysis.description.captions[0].text} (Confidence: {analysis.description.captions[0].confidence*100:.2f}%)")
    st.write("### Dominant Color Foreground:", analysis.color.dominant_color_foreground)
    st.write("### Dominant Color Background:", analysis.color.dominant_color_background)
    st.write("### Image Type:")
    st.write(f"  Clip Art Type: {analysis.image_type.clip_art_type}")
    st.write(f"  Line Drawing Type: {analysis.image_type.line_drawing_type}")

def create_prompt(system_message, messages):
    prompt = system_message
    message_template = "\n{}\n{}\n"
    for message in messages:
        prompt += message_template.format(message['sender'], message['text'])
    prompt += "\nassistant\n"
    return prompt

def create_human_readable_prompt(analysis):
    # Extract Key Information
    description = analysis.description.captions[0].text if analysis.description.captions else ""
    tags = ", ".join(tag.name for tag in analysis.tags)
    dominant_color_foreground = analysis.color.dominant_color_foreground
    dominant_color_background = analysis.color.dominant_color_background
    
    # Create a concise and human-readable prompt
    prompt = f"The picture has a description: {description}. "
    prompt += f"The dominant foreground color is {dominant_color_foreground}, and the dominant background color is {dominant_color_background}. "
    prompt += f"The tags associated with the picture are: {tags}. "
    prompt += "What is in the picture?"
    
    return prompt

def what_do_you_see(json_results, analysis):
    # Create a more concise and human-readable prompt for OpenAI
    prompt = create_human_readable_prompt(analysis)
    
    # Display the new prompt
    st.write("### New Prompt Sent to OpenAI:")
    st.write(prompt)
    
    # Get Response from OpenAI with the new prompt
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=prompt,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[""])
    
    # Display the entire response object to check if a valid response is received.
    st.write("### OpenAI API Response Object:")
    st.write(response)
    
    # Display Analysis and Response
    st.write("### Azure Computer Vision Semantic Analysis:")
    st.write("### OpenAI Response:")
    if response and response.choices:
        st.write(response.choices[0].text.strip())
    else:
        st.write("No response received from OpenAI.")

    
    # Display Analysis and Response
    st.write("### Azure Computer Vision Semantic Analysis:")
    st.write("### OpenAI Response:")
    if response and response.choices:
        st.write(response.choices[0].text.strip())
    else:
        st.write("No response received from OpenAI.")

       
    # Display Analysis and Response
    st.write("### Azure Computer Vision Semantic Analysis:")
    # st.write(json_results)
    st.write("### OpenAI Response:")
    st.write(response.choices[0].text.strip())

st.title('Dr. Lee Azure Semantic Vision App')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
url = st.text_input("Or enter Image URL:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    analysis = analyze_image(image)
    json_results = json.dumps(analysis.as_dict())
    what_do_you_see(json_results,analysis)
    display_analysis(analysis)

elif url:
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Image from URL.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            analysis = analyze_image(image)
            json_results = json.dumps(analysis.as_dict())
            what_do_you_see(json_results,analysis)
            display_analysis(analysis)
        else:
            st.error("The URL does not point to a valid image. Content-Type received was " + content_type)
            
    except requests.RequestException as e:
        st.error(f"Failed to fetch image due to request exception: {str(e)}")
        
    except requests.HTTPError as e:
        st.error(f"HTTP Error occurred: {str(e)}")
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
