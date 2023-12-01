#this currently uses streamlit as the UI / "Frontend"
import streamlit as st
from gnews import GNews
import nltk
from nltk.corpus import stopwords
import cohere
from dotenv import load_dotenv
import os

nltk.download('stopwords')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

def main():
    #Sets up aspects such as the streamlit website
    load_dotenv()
    st.set_page_config(page_title="Quick Report",
                       page_icon=":rolled_up_newspaper:")
    st.header("Quick Report :rolled_up_newspaper:\n")
    google_news = GNews()

    #prompts user for input
    subject = st.text_input('**Choose a Subject to Summarize:**')

    #code only continues once user presses enter to prevent error popup
    #(note: error popup only happens on initialization)
    if subject is not "":
        cleanSubj = removeStops(subject)

        news = getArticles(cleanSubj, google_news, subject)

        st.write("Found " + str(len(news)) + " articles on " + subject)
        
        newsSelection = presentOptions(news, google_news)

        if newsSelection is not None:
            displayText(newsSelection, google_news)

def removeStops(subject):
    with st.spinner("Processing Input..."):
        cachedStopWords = stopwords.words("english")
        return ' '.join([word for word in subject.split() if word not in cachedStopWords])

def getArticles(cleanSubj, google_news, subject):
    with st.spinner("Getting articles of \""+ subject +"\"..."):
        news = google_news.get_news(cleanSubj)
    return news

def presentOptions(news, google_news):
    pressed = False
    col1, col2, col3 = st.columns(3)
    with col1:
        if(st.button("Main Article")):
            newsSelection = getTopXArticles(1, news, google_news)
            pressed = True 
        if(st.button("Top 5 Articles")):
            newsSelection = getTopXArticles(5, news, google_news)
            pressed = True 
    with col2:
        if(st.button("Top 10 Articles")):
            newsSelection = getTopXArticles(10, news, google_news)
            pressed = True 
        if(st.button("Top 25 Articles")):
            newsSelection = getTopXArticles(25, news, google_news)
            pressed = True 
    with col3:
        if(st.button("Top 50 Articles")):
            newsSelection = getTopXArticles(50, news, google_news)
            pressed = True 
    if pressed:
        return newsSelection

def getTopXArticles(x, news, google_news):
    newsSelection = []
    for article in news:
        full_article = google_news.get_full_article(article['url'])
        if full_article is not None:
            if len(full_article.text) > 250:
                newsSelection.append(article)
                x -= 1
        if x == 0:
            return newsSelection


def displayText(newsSelection, google_news):
    i = 1
    newsSummary = ""
    images = []
    for article in newsSelection:
        with st.spinner("Summarizing Article #" + str(i) + "..."):
            st.write("**Article #" + str(i) + ": "+ article['title'] +"**")
            article_text = google_news.get_full_article(article['url']).text
            summary = summarizeText(article_text, 'medium')
            images.extend(google_news.get_full_article(article['url']).images)
            summaryProcessed = summary.replace("$", "\$")
            st.write(summaryProcessed)
            st.write(article['url'])
            st.write("------\n")
            newsSummary += summary
            i += 1
    if(i > 2):
        st.write("**In Summary:**")
        with st.spinner("Forming Summary of Topic..."):
            print(newsSummary)
            fullSum = summarizeText(newsSummary,'long')
            fullProcessed = fullSum.replace("$","\$")
            st.write(fullProcessed)

    # if len(images) > 0:
    #     rand = random.randint(0,len(images)-1)
    #     randImg = images[rand]
    #     urllib.request.urlretrieve(randImg, "img.jpg")
    #     img = Image.open("img.jpg")
    #     st.image(img)

    

def summarizeText(article, len):
    co = cohere.Client(COHERE_API_KEY)
    response = co.summarize(
        text=article,
        model = 'command-light-nightly',
        temperature = 0,
        length = len,
        format = 'paragraph',
        extractiveness = 'low',
        additional_command = "making sure to incorporate elements from all parts of the article and trying to connect them together"
    )
    return response.summary


main()