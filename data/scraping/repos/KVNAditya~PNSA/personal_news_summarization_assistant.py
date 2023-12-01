import streamlit as st
import time
import gnewsclient.gnewsclient as gnewsclient
import translators as ts

from langchain.document_loaders import NewsURLLoader

import nltk
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
        for itr in config__lc_nul.load():
            try:
                tle__lc_nul = 'Title : ' + itr.metadata['title']
                smry__lc_nul = 'Summary : ' + itr.metadata['summary']
                dspn__lc_nul = 'Description : ' + itr.metadata['description']
                if(0):
                    pass
                elif(len(smry__lc_nul)>1):
                    st.write(tle__lc_nul)
                    st.write(smry__lc_nul)
                    if(args__opt_lang in config__ts_langs):
                        tle__lc_nul = ts.translate_text(tle__lc_nul,from_language='auto',to_language=config__ts_langs[args__opt_lang])
                        st.write(tle__lc_nul)
                        smry__lc_nul = ts.translate_text(smry__lc_nul,from_language='auto',to_language=config__ts_langs[args__opt_lang])
                        st.write(smry__lc_nul)
                    else:
                        st.write(tle__lc_nul)
                        st.write(smry__lc_nul)
                    st.divider()
                elif(len(dspn__lc_nul)>1):
                    st.write(tle__lc_nul)
                    st.write(dspn__lc_nul)
                    if(args__opt_lang in config__ts_langs):
                        tle__lc_nul = ts.translate_text(tle__lc_nul,from_language='auto',to_language=config__ts_langs[args__opt_lang])
                        st.write(tle__lc_nul)
                        dspn__lc_nul = ts.translate_text(dspn__lc_nul,from_language='auto',to_language=config__ts_langs[args__opt_lang])
                        st.write(dspn__lc_nul)
                    else:
                        st.write(tle__lc_nul)
                        st.write(dspn__lc_nul)
                    st.divider()
                else:
                    pass
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

st.divider()
st.markdown("<h2 style='font-size: 28px; text-align: center; color: white;'>Personal News Summarization Assistant (PNSA)</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 16px; text-align: center; color: white;'>CMR Technical Campus | Surge Classes | Deep Learning | Lang Chain<h2>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 16px; text-align: center; color: white;'>K.V.N.Aditya | P.Sai Karthik | P.Phanindra | M.Venu | B.Lokesh Reddy<h2>", unsafe_allow_html=True)
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
