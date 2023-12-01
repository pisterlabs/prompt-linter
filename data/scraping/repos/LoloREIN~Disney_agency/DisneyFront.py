import joblib
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words_en = stopwords.words('English')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
import plotly_express as px
import re 
import streamlit as st
import pandas as pd
from PIL import Image
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage


#Importamos DataFrame
df = pd.read_csv(r'df_Binario.csv', encoding='ISO-8859-1')

#Importamos modelos de IA
modeloSVC = joblib.load(r'ModeloSVC.joblib')
modelRegresion = joblib.load(r'RegresionL3.joblib')
tfidf = joblib.load(r'tfidf_vectorizer3.joblib')

#Preprocesamiento de la información
def limpiar(texto):
    res = texto.lower()
    res = re.sub(r'[^a-zA-Z0-9\s]', '', res)
    res = word_tokenize(res)
    res = [lemmatizer.lemmatize(token) for token in res]
    res = [token for token in res if token not in stop_words_en]
    res = ' '.join(res)
    return res

def predecir_sentimiento_R(texto):
    texto_limpo = limpiar(texto)
    vectorizado = tfidf.transform([texto_limpo])
    prediccion = modelRegresion.predict(vectorizado)
    return prediccion[0]

# Agrupar el DataFrame por la columna 'Reviewer_Location' y contar las ocurrencias
df_grouped = df['Reviewer_Location'].value_counts().reset_index()
df_grouped.columns = ['Reviewer_Location', 'Count']

# Crear un mapa del mundo con Plotly Express
fig = px.choropleth(
    df_grouped,
    locations='Reviewer_Location',
    locationmode='country names',
    color='Count',
    hover_name='Reviewer_Location',
    title='Number of Reviews per country',
    color_continuous_scale=px.colors.sequential.Plasma,  # Puedes cambiar la escala de color según tu preferencia
)

# Establecer el fondo como transparente
fig.update_layout(
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showframe=False,
        showcoastlines=False,
        projection_type="natural earth",  # Puedes cambiar la proyección según tu preferencia
    )
)

# Agrupar el DataFrame por 'Year' y 'Branch' y contar las ocurrencias
df_grouped = df.groupby(['Year', 'Branch']).size().reset_index(name='Count')

# Crear la gráfica de barras
bar_year = px.bar(
    df_grouped,
    x='Year',
    y='Count',
    color='Branch',
    color_discrete_sequence=px.colors.qualitative.Dark24,
    title='Total Reviews per Year and Park',
    labels={'Count': 'Total Reviews', 'Year': 'Año'},
)

# Agrupar el DataFrame por 'Sentiment' y contar las ocurrencias
df_sentiment_count = df['Sentiment'].value_counts().reset_index()
df_sentiment_count.columns = ['Sentiment', 'Count']

# Crear la gráfica de barras
bar_sent = px.bar(
    df_sentiment_count,
    x='Sentiment',
    y='Count',
    color='Sentiment',
    color_discrete_sequence=px.colors.qualitative.G10,
    title='Total Reviews',
    labels={'Count': 'Total Reviews', 'Sentiment': 'Sentimiento'},
)



# texto = '''
# I love disney and I had a great time through out the day unfortunely when my boyfriend and I were leaving we got stopped by a working that was saying we looked suspicious? How do we look suspicious is it because we are latin? I mean we were already outside of the park when he called us back in to do a random check? What??? we are leaving home why do a check when you should of checked for weapons when we were about to enter the park not after!! Racist much? It didn't bother me until he would let all white people pass by but would stop hispanics WOW!! I had a wallet idk what weapon would of fit in there but now I regret not leaving and denying to reenter the park just for a "random check". Never going back, I can't believe they will hire such racist people.
# '''
# predictionLR = predecir_sentimiento_R(texto)
# predictionSVC = modeloSVC.predict([texto])
# predictionSVC = modelofnn.predict([texto])
# Print the prediction
# print(f'According to the Logistic Regression model, the review is: {predictionLR}')
# print(f'According to the SVC model, the review is: {predictionSVC[0]}')

bigram_Good = Image.open(r'Bigramas_Good_Binario.png')
Trigram_Good = Image.open(r'Trigramas_Good_Binario.png')
bigram_Bad = Image.open(r'Bigramas_Bad_Binario.png')
Trigram_Bad = Image.open(r'Trigramas_Bad_Binario.png')

st.markdown("<h1 style='text-align: center; color: white;'>Disneyland Adventure Agency</h1>", unsafe_allow_html=True)
# st.write(df.head(5))
st.markdown("<h2 style='text-align: center; color: white;'>Binary Classification</h2>", unsafe_allow_html=True)
st.write(bar_sent)
st.write(fig)
st.write(bar_year)
col1,col2 = st.columns(2)
col1.write('### Bigrams')
col1.write('### \n ')

# Organizar las imágenes en una fila en la primera columna
col1.image(bigram_Good)
col1.image(bigram_Bad)

col2.write('### Trigrams')
col2.write('### \n ')

# Organizar las imágenes en una fila en la segunda columna
col2.image(Trigram_Good)
col2.image(Trigram_Bad)

st.write('### Say if a review is good or bad')
texto = st.text_input("Enter a review:")
predictionLR = predecir_sentimiento_R(texto)
predictionSVC = modeloSVC.predict([texto])

if (texto):
    with st.spinner('Esperate que ando chambeando...'):
        st.write(f'This is a {predictionLR} review. According to Logistic Regresion model')
        st.write(f'This is a {predictionSVC[0]} review. According to SVC model')


st.markdown("<h2 style='text-align: center; color: white;'>Multi-class Classification</h2>", unsafe_allow_html=True)

bigram_P = Image.open(r'Bigramas_Positive.png')
Trigram_P = Image.open(r'Trigramas_Positive.png')
bigram_Neg = Image.open(r'Bigramas_Negative.png')
Trigram_Neg = Image.open(r'Trigramas_Negative.png')
bigram_Neu = Image.open(r'Bigramas_Neutral.png')
Trigram_Neu = Image.open(r'Trigramas_Neutral.png')

st.markdown("<h2 style='text-align: center; color: white;'>Multi-class Classification</h2>", unsafe_allow_html=True)

col3,col4 = st.columns(2)
col3.write('### Bigrams')
col3.write('### \n ')

# Organizar las imágenes en una fila en la primera columna
col3.image(bigram_P)
col3.image(bigram_Neg)
col3.image(bigram_Neu)

col4.write('### Trigrams')
col4.write('### \n ')

# Organizar las imágenes en una fila en la segunda columna
col4.image(Trigram_P)
col4.image(Trigram_Neg)
col4.image(Trigram_Neu)

st.write('### Say if a review is good or bad')
texto = st.text_input("Enter a review:")
#AQUI ESCRIBE EL METODO PARA PREDECIR
if (texto):
    with st.spinner('Esperate que ando chambeando...'):
        st.write(f'This is a {predictionLR} review. According to CNN model')


with st.sidebar:
    api_key_file = st.file_uploader("Upload your key",type=["txt"])
    if api_key_file is not None:
        key = str(api_key_file.readline().decode("utf-8"))
        os.environ["OPENAI_API_KEY"]=key
        llm = ChatOpenAI(model_name = "gpt-3.5-turbo")

        st.write("We know disney can be challenging")
        query = st.text_input("So ask us anything :)")
        promt = '''
        You are a travel assistant who will friendly answer questions from those interested in traveling to Disney and learning about its parks. Includes details on attractions, special events, park visiting strategies, news and updates at Disney parks around the world. Avoid including information that is not directly related to Disney or its theme parks. And finish all of your answers with "and remember Disneyland is where the dreams come true."
                '''

        if st.button("generate output"):
            response = llm.invoke([SystemMessage(content=promt),HumanMessage(content=query)])
            st.write(f'##### {response.content}')
        llm = ChatOpenAI(model_name = "gpt-3.5-turbo")

    
