# Import required libraries
import os
import io
import utils
import streamlit as st
from PIL import Image, ImageDraw
import requests
from langchain.agents import create_sql_agent, AgentExecutor, load_tools, AgentType, initialize_agent
from langchain.llms import OpenAI
import openai
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from streaming import StreamHandler
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import sqlalchemy
from bot_instructions import chatbot_instructions, sqlbot_instructions
from langchain.chains import LLMChain, SequentialChain

# Streamlit page setup
st.set_page_config(page_title="GenAg Image Asset Generator", page_icon="🌱") #, layout="wide")
st.subheader("Image Asset Generator 🌱📸")
st.write("""
**Turn your physical farm assets into digital assets for farm mapping.**

**How to Use:**
- Take pictures of your physical farm assets.
- Upload them through the interface below.

**Note:** This tool is a proof of concept and may not be perfect. We're continually improving it.
""")

# Define the main class for the Generative Agriculture Chatbot
class GenerativeAgriculture:
    # Initialize chatbot settings and API keys
    def __init__(self):
        utils.configure_openai_api_key()
        #self.openai_model = "gpt-3.5-turbo-instruct"
        #self.openai_model = "gpt-3.5-turbo"
        self.openai_model = "gpt-4-0613"
        #self.openai_model = "gpt-4-32k" # 4x context length of gpt-4
    
    # Setup database and agent chain
    @st.cache_resource
    def setup_chain(_self):
        # Database Connection
        username, password, host, port, database = [st.secrets[key] for key in ["username", "password", "host", "port", "database"]]
        db_url = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
        db = SQLDatabase.from_uri(db_url)

        # Initialize Database Connection
        try:
            engine = sqlalchemy.create_engine(db_url)
            conn = engine.connect()
        except Exception as e:
            st.write(f"An error occurred: {e}")
            exit()
        
        # Initialize memory setup (commented out for future use)
        chatbot_memory = None
        # sqlagent_memory = ConversationBufferMemory()

        chatbot_instructions = ""

        # Setup Chatbot
        chatbot_prompt_template = PromptTemplate(
            input_variables = ['task_list', 'user_input'],
            template=chatbot_instructions
        )
        llm=OpenAI(model_name=_self.openai_model, temperature=0.0, streaming=True)
        chatbot_agent = LLMChain(
            llm=llm, 
            memory=chatbot_memory, 
            prompt=chatbot_prompt_template, 
            verbose=True)
        
        return chatbot_agent
    
    # Main function to handle user input and chatbot response
    # @utils.enable_chat_history
    def main(self):

        # DALL-E Image Transformation
        st.subheader("Transform Your Image with DALL-E")
        uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Convert the uploaded image to 'RGBA' format
            image = Image.open(uploaded_image).convert("RGBA")
            
            # Generate a mask that matches the dimensions of the uploaded image
            mask = Image.new("L", image.size, 0)  # Start with a fully black mask
            draw = ImageDraw.Draw(mask)
            start_x = int(image.size[0] * 0.375)  # 37.5% of width
            start_y = int(image.size[1] * 0.375)  # 37.5% of height
            end_x = int(image.size[0] * 0.625)  # 62.5% of width
            end_y = int(image.size[1] * 0.625)  # 62.5% of height
            draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=255)
            
            # Turn the converted image and mask back into byte streams
            image_bytes_io = io.BytesIO()
            mask_bytes_io = io.BytesIO()
            
            image.save(image_bytes_io, format='PNG')
            mask.save(mask_bytes_io, format='PNG')
            
            image_bytes = image_bytes_io.getvalue()
            mask_bytes = mask_bytes_io.getvalue()

            # Default DALL-E prompt
            default_prompt = "an 8-bit retro-style digital asset viewable from a top-down perspective that could be used in the game Farmville."
            
            # Allow users to edit the prompt
            user_prompt = st.text_input("DALL-E Prompt:", default_prompt)
            
            if st.button('Transform Image'):
                # Run DALL-E API call
                response = openai.Image.create_edit(
                    image=image_bytes,
                    mask=mask_bytes,
                    prompt=user_prompt,
                    n=1,
                    size="1024x1024"
                )
                
                image_url = response['data'][0]['url']
                st.image(image_url, caption="Transformed Image", use_column_width=True)


                
# Entry point of the application
if __name__ == "__main__":
    obj = GenerativeAgriculture()
    obj.main()
