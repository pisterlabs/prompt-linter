'''Instrucciones como utilizar este codigo en la PC (no se puede ejecutar desde COLAB)

1 - Debemos tener instalado en la PC python y Anaconda o visual studio code.
2 - Ejecutar desde la consola: pip install streamlit
3 - Navegar desde la consola hasta la carpeta en donde se encuentra este codigo en la PC.Ejecutar este codigo
4 - Si tira error de no se encuentra la libreria Streamlit.cli debemos reinstalar streamlit usando:
    1- pip unistall streamlit
    2- pip install streamlit
5 - Una vez ejecutado se abrirá una pagina web desde chrome con nuestra APP que se encuentra local.
'''

import subprocess

# Verificar si la biblioteca está instalada
try:
    import matplotlib.pyplot as plt
except ImportError:
    # La biblioteca no está instalada, se procede a instalarla
    subprocess.check_call(['pip', 'install', 'matplotlib'])
    
import string
import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import lightgbm as lgb
import openai

# Desactivar la advertencia de usar pyplot global
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("✨Amazon Product Title Accuracy Specialist Web App")
    st.markdown("Are you sure about your Descripction Product? Let me help you review it and giving you my professional suggestion")

    @st.cache_resource()
    def load_LGBM():
        with open('modelo_LGBM.pkl', 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    
    @st.cache_resource()
    def load_vectorizer():
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            modelo = pickle.load(f)
        return modelo
      
    # Mostrar el texto ingresado
    def show_input_text(input_text):
        st.write("The input text is: ", input_text)

    @st.cache_resource()
    def normalize(text):
        text = text.lower()  # Convert text to lowercase
        tokens = word_tokenize(text)  # Tokenize the text into words
        tokens = [token for token in tokens if token.isalnum() and token not in stopwords_en]  # Remove stop words
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_text = ' '.join(lemmas)
        return preprocessed_text
    
    # Main
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    ## Load Modules
    #lemmatizer  = WordNetLemmatizer()
    stopwords_en   = set(nltk.corpus.stopwords.words('english'))
    punctuation = string.punctuation
    stemmer = PorterStemmer() 

    input_text = st.text_input("Write here your Description Product", "")  # load text input  

    # Predict if the title is good or not using LGBM 
    if input_text != "":
        show_input_text(input_text) # show input text
        X_clean = normalize(input_text) # preprocess text
        modelo_vectorizer = load_vectorizer() # Convert the preprocessed text into a TF-IDF vector
        df_vectors_test = modelo_vectorizer.transform([X_clean]) # Predict using imported PKL vectorizer model
        modelo_LGBM = load_LGBM()  #Load model
        predictions = modelo_LGBM.predict(df_vectors_test) # Predict using imported PKL LGBM model
        predictions = predictions[0]
    
        threshold = 0.05
        if predictions > threshold: 
          predictions = 1 
        else: predictions = 0
    
        # Print the prediction
        if predictions == 1:
            show_input_text("The Description is Awesome") # show input text
        else:
            show_input_text("This Description is not so good....") # show input text
            
    # Give a reviewed description of the product using chatGPT 4
    openai_key = st.text_input("Write here your Open AI Key", "")  # load openaikey 
    openai.api_key = openai_key
    if openai_key != "":
        concent = 'You are a Margeting redactor content specialist. Your goal is redo the product Description that I give you.\
        Do a short text without any aditionally explanation about it or any comment. I need you to improve this description to get more sales'
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
              {"role": "system", "content": concent},
              {"role": "user", "content": input_text}
            ],
            temperature=.5,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        prompt_response = response["choices"][0]["message"]['content'].strip()
        request = 'Description Product: ' + prompt_response
        st.text(request)
    
if __name__ == '__main__':
    main()




