import streamlit as st
import time
import gnewsclient.gnewsclient as gnewsclient
import nltk
import tempfile
import os

from googletrans import Translator
from gtts import gTTS
from langchain.document_loaders import NewsURLLoader

nltk.download('punkt')

def func__init__gnc_lc_ts(args__mtch_btn):
    op_log = st.empty()
    op_log.text("connecting to GoogleNewsAPI ...")
    time.sleep(2)
    op_log.text("successfully connected to GoogleNewsAPI ...")
    time.sleep(2)
    op_log.text("fetching news ...")
    time.sleep(2)
    op_log.text("summarizing the news extracted from the urls ...")
    time.sleep(2)
    op_log.text("translating the summarized news results ...")
    time.sleep(2)
    op_log.text("returning the translated news results ...")
    time.sleep(2)
    op_log.empty()
    time.sleep(2)
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
        for itr in enumerate(config__lc_nul.load()):
            try:
                cls__gT = Translator()
                tle__lc_nul_gT,dspn__lc_nul_gT,smry__lc_nul_gT = '','',''
                str__tle_despn_smry = ''
                
                if((len(itr[1].metadata['title']) != 0)):
                    tle__lc_nul = 'Title : ' + itr[1].metadata['title']
                    tle__lc_nul_gT = cls__gT.translate(tle__lc_nul, dest=config__ts_langs[args__opt_lang]).text
                    str__tle_despn_smry += str('.' + tle__lc_nul_gT + '.')

                if((len(itr[1].metadata['description']) != 0)):
                    dspn__lc_nul = 'Description : ' + itr[1].metadata['description']
                    dspn__lc_nul_gT = cls__gT.translate(dspn__lc_nul, dest=config__ts_langs[args__opt_lang]).text
                    str__tle_despn_smry += str('.' + dspn__lc_nul_gT + '.')

                if((len(itr[1].metadata['summary']) != 0)):
                    smry__lc_nul = 'Summary : ' + itr[1].metadata['summary']
                    smry__lc_nul_gT = cls__gT.translate(smry__lc_nul, dest=config__ts_langs[args__opt_lang]).text
                    str__tle_despn_smry += str('.' + smry__lc_nul_gT + '.')
                
                gTTS__str_tle_despn_smry = gTTS(str__tle_despn_smry,lang=config__ts_langs[args__opt_lang])
                tmpf__gTTS_str_tle_despn_smry = tempfile.NamedTemporaryFile(suffix='.wav',delete=False)
                gTTS__str_tle_despn_smry.save(tmpf__gTTS_str_tle_despn_smry.name)
                tmpf__gTTS_str_tle_despn_smry.close()
                
                st.markdown(f"[{tle__lc_nul_gT}]({args__ul__gnc_nc[itr[0]]})")
                st.audio(tmpf__gTTS_str_tle_despn_smry.name)
                st.write(dspn__lc_nul_gT)
                st.write(smry__lc_nul_gT)

                if(itr[0] < len(args__ul__gnc_nc)-1):
                    st.subheader('',divider='green')
                
            except Exception as e:
                st.write(e)

                
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

st.subheader('',divider='rainbow')
st.markdown("<p style='font-size: 28px; text-align: center; color: lightgreen !important;'><a href='#' style='color: lightgreen;'><u>Personal News Summarization Assistant (PNSA)</u></a></p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; text-align: center; color: lightblue;'>||&emsp;CMR Technical Campus&emsp;|&emsp;Surge Classes&emsp;|&emsp;Deep Learning&emsp;|&emsp;Lang Chain&emsp;||</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; text-align: center; color: lightyellow;'>~&emsp;K.V.N.Aditya&emsp;*&emsp;P.Sai Karthik&emsp;*&emsp;P.Phanindra&emsp;*&emsp;M.Venu&emsp;*&emsp;B.Lokesh Reddy&emsp;~</p>", unsafe_allow_html=True)
st.subheader('',divider='rainbow')
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size:15px; color: deeppink;'>!!! personalize your news feed !!!</h1>", unsafe_allow_html=True)
    st.subheader('',divider='rainbow')
    st_sb_opt_loc = st.selectbox('Choose Location', lst_gnc_nc_locs,help="opt a location ...",placeholder="choose a location",index=None)
    st_sb_opt_tpc = st.selectbox('Choose Topic', lst_gnc_nc_tpcs,help="opt a topic ...",placeholder="choose a topic",index=None)
    st_sb_opt_lang = st.selectbox('Choose Language', lst_gnc_nc_langs,help="opt a language ...",placeholder="choose a language",index=None)
    st_sb_opt_nc = st.select_slider('Choose News Count', range(1,21,1),value=2)
    st.subheader('',divider='rainbow')
    st_sb_btn_cols = st.columns(2)
    with st_sb_btn_cols[0]:
        st_sb_btn_gns = st.button("Get News Summarization",key=0)
    with st_sb_btn_cols[1]:
        st_sb_btn_gnm = st.button("Get News MetaData",key=1)

if(st_sb_btn_gns):
    func__init__gnc_lc_ts(args__mtch_btn=0)
if(st_sb_btn_gnm):
    func__init__gnc_lc_ts(args__mtch_btn=1)