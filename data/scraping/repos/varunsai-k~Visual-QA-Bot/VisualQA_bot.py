import streamlit as st
from PIL import Image
import io
import os
from transformers import ViltProcessor, ViltForQuestionAnswering
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

os.environ["OPENAI_API_KEY"] = 'Your API Key here'

st.set_page_config(page_title="Visual QA Bot",page_icon="🤖")

st.title('Visual Q:red[/]A Bot🤖')
st.write("Explore the power of visual communication as our cutting-edge AI engages in conversations with your images. Ask about scenes, objects, or anything you're curious about, and let our bot provide insightful responses.")
bot = st.chat_message("assistant")
bot.write("""👋 Welcome to Visual Q:red[/]A Bot!\n
We're thrilled to have you here. To embark on this visual adventure, simply upload an image that piques your curiosity, and let the conversation unfold!

📷How it works:

    1.Upload Image: Share any image from your gallery or take a new snapshot.
    2.Ask Your Questions: Wondering about objects, scenes, or details? Just ask, and our bot will analyze the visual content.
    3.Enjoy the Conversation: Engage in a natural, chat-like experience as our AI decodes the intricacies of your image.

""")
uploaded_file = st.file_uploader("Choose a image", accept_multiple_files=False)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
def generate_answer(question,answer):
    template="""
    You are a Question Answer Bot;You have to generate a conversational response by understanding the question along with one word answer and rewrite the answer in a creative way;
    
    QUESTION: {question}
    
    ANSWER: {answer}

    Response: 
    """
    answer_prompt=PromptTemplate(template=template, input_variables=["question","answer"])
    
    answer_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=answer_prompt,verbose=True)
    response=answer_llm.predict(question=question,answer=answer)
    return response
        
        
if uploaded_file:
    data=uploaded_file.read()
    #data=data.resize((600, 500), Image.ANTIALIAS)
    image_buffer = io.BytesIO(data)
    pil_image = Image.open(image_buffer)
    converted_image = pil_image.convert('RGB')

    with st.columns(3)[1]:
        st.image(data,width=500)
        
    if prompt := st.chat_input("Ask your Questions"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # prepare inputs
            encoding = processor(converted_image, prompt, return_tensors="pt")

            # forward pass
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = f"{model.config.id2label[idx]}"
            response=generate_answer(prompt,answer)
            #st.write(f"Model Output: {answer}")
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        
else:
    if prompt := st.chat_input("Ask your Questions"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = """Hello, 🌟Great Move! Ready for Visual Exploration!

Before we dive into questions, let's add a splash of visuals to our conversation. Please upload an image that catches your interest or sparks your curiosity. Once the image is in, feel free to ask anything about it, and together we'll unravel the stories within.

Excited to see what you've got! 📸🚀
        
        """
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    
if st.button("Clear Chat History"):
    st.session_state.messages.clear()
