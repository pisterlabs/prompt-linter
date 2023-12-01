from urllib.request import urlopen
from bs4 import BeautifulSoup
import streamlit as st
import re
import html2text
import openai
import thread_content
#from decouple import config

if "data_" not in st.session_state:
    st.session_state.data_ = thread_content.data_

if "submit_thread" not in st.session_state:
    st.session_state.submit_thread = False

def parse_html(thread_id):
    if thread_id == "Thread 1":
        html = thread_content.thread_1 + thread_content.thread_2
    else:
        html = thread_content.thread_2
    h = html2text.HTML2Text()
    h.ignore_links = True
    text =  h.handle(html)
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def structure_text(text):
    structured_text = []
    split_text = re.split(r'\*\s\*\s\*', text)
    for i in split_text:
        thread = {}
        user_name = i.split("\n")[1]
        if "https" in user_name:
            user_name = user_name.split()[-2] + user_name.split()[-1]
        user_name = user_name.replace("_","")
        if ")" in user_name:
            user_name = user_name.split(")")[-1]
        thread["UserName"] = user_name
        if len((i.split("\n")[2]).split(" ")) > 1:
            thread["Date"] = i.split("\n")[2]
        else:
            thread["Date"] = i.split("\n")[3]
        if "behalf" in thread["Date"] or "*" in thread["Date"]:
            continue
        thread["Thread_id"] = i.split("\n")[3]
        sp = "\n".join(i.split("\n")[4:])
        if len(sp.split("toggle quoted messageShow quoted text")) > 1:
            thread["content"] = re.sub(r'[^a-zA-Z :\.\d]',"",sp.split("toggle quoted messageShow quoted text")[0])
            thread["quoted"] = re.sub(r'[^a-zA-Z :\.\d]',"",sp.split("toggle quoted messageShow quoted text")[1])
        else:
            thread["content"] = re.sub(r'[^a-zA-Z :\.\d]',"",sp)
            thread["quoted"] = ""
        if len(user_name) > 1:
            structured_text.append(thread)

    return structured_text, split_text

def insight(chunk):
    start_sequence = "The main topic of conversation in under 8 words is:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="\""+chunk+"\"" +"\n"+start_sequence,
        temperature=0.7,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    insight = response.choices[0].get("text")
    insight = insight.replace("\"", "")
    return insight.strip()


def thread_summary(chunk):
    start_sequence = "\nA 100 word summary:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="\""+chunk+"\"" +"\n"+start_sequence,
        temperature=0.7,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    insight = response.choices[0].get("text")
    insight = insight.replace("\"", "")
    return insight.strip()


def heading(chunk):
    start_sequence = "\nA title for the convaersation is:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="\""+chunk+"\"" +"\n"+start_sequence,
        temperature=0.7,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    insight = response.choices[0].get("text")
    insight = insight.replace("\"", "")
    return insight.strip()



def reply_summary(chunk):
    start_sequence = "\nTl;dr:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="\""+chunk+"\"" +"\n"+start_sequence,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    insight = response.choices[0].get("text")
    insight = insight.replace("\"", "")
    return insight.strip()


def key_points(chunk):
    start_sequence = "\nThe main points of the conversation are:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="\""+chunk+"\"" +"\n"+start_sequence,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    insight = response.choices[0].get("text")
    insight = insight.replace("\"", "")
    return insight.strip()


def get_thread_content(struct_text):
    content = ""
    for replies in struct_text:
        content += re.sub(r'http\S+', '', replies["content"]) + "\n"
    return content

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Email Threads Summarization Demo")
    #openai.api_key = config("OPENAI_KEY")
    openai.api_key = st.secrets["OPENAI_KEY"]
    st.write("""---""")
    text = parse_html("Thread 1")
    struct_text, split_text = structure_text(text)
    thread_content = get_thread_content(struct_text)
    if len(thread_content.split(" ")) > 2500:
        thread_content = " ".join((thread_content.split(" "))[0:2500])

    st.header("Threads Insight")
    st.write("""---""")
    with st.container():
        cols = st.columns([1,0.2,5])
        cols[0].subheader("Title")
        cols[1].subheader(":")
        #cols[2].info("test")
        if st.session_state.data_["title"] == "":
            init_heading = heading(thread_content)
            st.session_state.data_["title"] = init_heading
        cols[2].info(st.session_state.data_["title"])

    with st.container():
        cols = st.columns([1,0.2,5])
        cols[0].subheader("Abstract")
        cols[1].subheader(":")
        #cols[2].info("test")
        if st.session_state.data_["abstract"] == "":
            init_abstract = thread_summary(thread_content)
            st.session_state.data_["abstract"] = init_abstract
        cols[2].info(st.session_state.data_["abstract"])
    
    with st.container():
        cols = st.columns([1,0.2,5])
        cols[0].subheader("Key Points")
        cols[1].subheader(":")
        #cols[2].info("test")
        if st.session_state.data_["keypoints"] == "":
            init_keypoints = key_points(thread_content)
            st.session_state.data_["keypoints"] = init_keypoints
        cols[2].info(st.session_state.data_["keypoints"])

    if len(st.session_state.data_["struct_text"]) == 0:
        count = 0
        new_struct_list_with_summary = []
        for i in range(len(struct_text)):
            with st.spinner("Processing..."+str(count)+"/"+str(len(struct_text))):
                curr_thread = struct_text[i]
                #insight = insight(curr_thread["content"]))
                if len(curr_thread["content"].split(" ")) >= 30:
                    summary = reply_summary(curr_thread["content"])
                else:
                    summary = "Content Too Short To Summarize..."
                curr_thread["summary"] = summary
                new_struct_list_with_summary.append(curr_thread)
                count += 1
        st.session_state.data_["struct_text"] = new_struct_list_with_summary

    st.write("""---""")
    with st.expander("Click here to view all summaries..."):
        for i in range(len(st.session_state.data_["struct_text"])):
            curr_thread = st.session_state.data_["struct_text"][i]
            with st.container():
                cols = st.columns([1,0.2,6])
                cols[0].text(curr_thread["UserName"])
                cols[1].text(":")
                if "Content Too Short To Summarize..." in curr_thread["summary"]:
                    cols[2].warning(curr_thread["summary"])
                else:
                    cols[2].caption(curr_thread["summary"])
                st.write("""---""")

    st.write("""---""")
    with st.container():
        cols = st.columns([1,5])
        cols[0].header("Date")
        cols[1].header("Thread Insights")
    st.write("""---""")

    for i in range(len(st.session_state.data_["struct_text"])):
        curr_thread = st.session_state.data_["struct_text"][i]
        with st.container():
            cols = st.columns([1,5])
            cols[0].text(curr_thread["Date"])
            with cols[1].expander(curr_thread["UserName"]):
                st.write("""---""")
                if st.checkbox("Show Thread content", key=i):
                    st.header("Parsed Thread Contents")
                    st.subheader("Thread ID:")
                    st.caption(curr_thread["Thread_id"])
                    st.subheader("Content:")
                    st.caption(curr_thread["content"])
                    if curr_thread["quoted"] != "":
                        st.subheader("Quoted:")
                        st.caption(curr_thread["quoted"])
                st.subheader("Summary :")
                if "Content Too Short To Summarize..." in curr_thread["summary"]:
                    st.warning(curr_thread["summary"])
                else:
                    st.info(curr_thread["summary"])
    st.write("""---""")
