import streamlit as st
import time
import gnewsclient.gnewsclient as gnewsclient
from langchain.document_loaders import NewsURLLoader
import nltk
nltk.download('punkt')
from googletrans import Translator
import pyshorteners
import tempfile
from gtts import gTTS
import os

def short_url(original_url):
    # Create a Shortener object
    s = pyshorteners.Shortener()
    try:
        # Shorten the URL
        shortened_url = s.tinyurl.short(original_url)

        # Display the shortened URL
        return shortened_url
    except pyshorteners.exceptions.ShorteningErrorException as e:
        # Handle the URL shortening error
        print(f"An error occurred while loading the URL")
    except Exception as e:
        # Handle other exceptions
        print(f"An unexpected error occurred")

def do_tts(text,language):
    tts = gTTS(text,lang=language)
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(temp_file.name)
    
    temp_file.close()
    return temp_file.name

def Audio(text,lang,key):
    st.subheader("Text-to-Speech Converter")

    # Input text
    text = text
    language = lang
    # st.write("This is not cool.")
    # if st.button("Convert to Speech",key=key):
    if True:
        st.write("Plase wait until it converts to speech")
        if text:
            file_path = do_tts(text,language)
            st.audio(file_path, format="audio/mp3")
            os.remove(file_path)

        else:
            st.warning("Please enter text before converting.")

    st.write("Powered by gTTS (Google Text-to-Speech)")

def translate_text_with_googletrans(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


def func__init__gnc_lc_ts(args__mtch_btn):
    # op_log = st.empty()
    # op_log.text("connecting to GoogleNewsAPI ...")
    # time.sleep(0.5)
    # op_log.text("successfully connected to GoogleNewsAPI ...")
    # time.sleep(0.5)

    # op_log.text("summarizing the news extracted from the urls ...")
    # time.sleep(2)
    # op_log.text("translating the summarized news results ...")
    # time.sleep(2)
    # op_log.text("returning the translated news results ...")
    # time.sleep(2)
    # op_log.empty()
    # time.sleep(2)
    func__lc_ts(func__gnc(st_sb_opt_loc,st_sb_opt_tpc,st_sb_opt_nc),st_sb_opt_lang,args__mtch_btn)

def func__gnc(args__opt_loc,args__opt_tpc,args__opt_nc):
    config__gnc_nc = gnewsclient.NewsClient(location=args__opt_loc,topic=args__opt_tpc,max_results=args__opt_nc)
    lst__ul__gnc_nc = [] # ul : url - links
    for itr_nc in range(args__opt_nc):
        try:
            lst__ul__gnc_nc.append(config__gnc_nc.get_news()[itr_nc]['link'])
        except:
            pass
    return lst__ul__gnc_nc

def func__lc_ts(args__ul__gnc_nc,args__opt_lang,args__mtch_btn):
    config__ts_langs = {'english' : 'en','telugu' : 'te','hindi' : 'hi'}
    config__lc_nul = NewsURLLoader(args__ul__gnc_nc,nlp=True)
    if(args__mtch_btn==0):
        i = 0
        for itr in config__lc_nul.load():
            try:
                tle__lc_nul = 'Title : ' + itr.metadata['title']
                smry__lc_nul = 'Summary : ' + itr.metadata['summary']
                dspn__lc_nul = 'Description : ' + itr.metadata['description']
                if(0):
                    pass
                elif(len(smry__lc_nul)>1):

                    # op_log = st.empty()
                    # op_log.text("fetching news ...")
                    # time.sleep(0.5)
                    # op_log.empty()
                    # time.sleep(0.5)

                    st.write(tle__lc_nul)
                    st.write(smry__lc_nul)
                    Audio(smry__lc_nul,'en',key=i+3)


                    st.write("Link for above News/Article: ",short_url(args__ul__gnc_nc[i]))

                    # op_log.text("summarizing the news extracted from the urls ...")
                    # time.sleep(0.5)

                    # op_log.empty()
                    # time.sleep(0.5)
                    if args__opt_lang!='english':
                        if(args__opt_lang in config__ts_langs):
                            # op_log = st.empty()
                            
                            # op_log.text("translating the summarized news results ...")
                            # time.sleep(0.5)

                            # op_log.empty()
                            # time.sleep(0.5)

                            tle__lc_nul = translate_text_with_googletrans(tle__lc_nul,config__ts_langs[args__opt_lang])
                            st.write(tle__lc_nul)
                            smry__lc_nul = translate_text_with_googletrans(smry__lc_nul,config__ts_langs[args__opt_lang])
                            st.write(smry__lc_nul)
                            if args__opt_lang == 'telugu':
                               Audio(smry__lc_nul,'te',key=i+3)
                            else:
                                Audio(smry__lc_nul,'hi',key=i+3)
                        else:
                            st.write(tle__lc_nul)
                            st.write(smry__lc_nul)
                    # st.divider()
                    st.markdown("<hr style='border: 2px solid #FF4B4B;'>", unsafe_allow_html=True)
                elif(len(dspn__lc_nul)>1):
                    st.write(tle__lc_nul)
                    st.write(dspn__lc_nul)
                    if args__opt_lang!='english':
                        if(args__opt_lang in config__ts_langs):
                            tle__lc_nul = translate_text_with_googletrans(tle__lc_nul,config__ts_langs[args__opt_lang])
                            st.write(tle__lc_nul)

                            dspn__lc_nul = translate_text_with_googletrans(dspn__lc_nul,config__ts_langs[args__opt_lang])
                            st.write(dspn__lc_nul)
                            if args__opt_lang == 'telugu':
                               Audio(dspn__lc_nul,'te',key=i+3)
                            else:
                                Audio(dspn__lc_nul,'hi',key=i+3)
                        else:
                            st.write(tle__lc_nul)
                            st.write(dspn__lc_nul)
                    # st.divider()
                    st.markdown("<hr style='border: 2px solid #FF4B4B;'>", unsafe_allow_html=True)
                else:
                    pass
            except Exception as e:
                st.write(e)
            i = i+1
    if(args__mtch_btn==1):
        for itr in config__lc_nul.load():
            try:
                st.write(itr.metadata)
            except Exception as e:
                st.write(e)


config__gnc_nc = gnewsclient.NewsClient()
lst_gnc_nc_locs = config__gnc_nc.locations
lst_gnc_nc_tpcs = config__gnc_nc.topics
lst_gnc_nc_langs = config__gnc_nc.languages
lst_gnc_nc_langs = ['english','telugu','hindi']

st.divider()
st.markdown("<h2 style='font-size: 32px; text-align: center; color: orange;'>Personal News Summarization Assistant (PNSA)</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 18px; text-align: center; color: white;'>CMR Technical Campus | Surge Classes | Deep Learning | Lang Chain<h2>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 18px; text-align: center; color: white;'>K.V.N.Aditya | P.Sai Karthik | P.Phanindra | M.Venu | B.Lokesh Reddy<h2>", unsafe_allow_html=True)
st.divider()
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size:15px; color: white;'>!!! opt the news based on your interests !!!</h1>", unsafe_allow_html=True)
    st.divider()
    st_sb_opt_loc = st.selectbox('Choose Location', lst_gnc_nc_locs,help="opt a location ...",placeholder="choose a location",index=None)
    st_sb_opt_tpc = st.selectbox('Choose Topic', lst_gnc_nc_tpcs,help="opt a topic ...",placeholder="choose a topic",index=None)
    st_sb_opt_lang = st.selectbox('Choose Language', lst_gnc_nc_langs,help="opt a language ...",placeholder="choose a language",index=None)
    st_sb_opt_nc = st.select_slider('Choose News Count', range(1,21,1),value=2)
    st.divider()
    st_sb_btn_cols = st.columns(2)
    with st_sb_btn_cols[0]:
        st_sb_btn_gns = st.button("Get News Summarization",key=0)
    with st_sb_btn_cols[1]:
        st_sb_btn_gnm = st.button("Get News MetaData",key=1)

if(st_sb_btn_gns):
    func__init__gnc_lc_ts(args__mtch_btn=0)
if(st_sb_btn_gnm):
    func__init__gnc_lc_ts(args__mtch_btn=1)
