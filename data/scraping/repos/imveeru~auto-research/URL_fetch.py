##################### Load URLs to LLM #####################

from langchain.document_loaders import UnstructuredURLLoader

def get_content_from_URLs(urls):
    loader=UnstructuredURLLoader(urls=urls)
    return loader.load()

content=get_content_from_URLs(urls)
st.write(content)

from langchain.text_splitter import CharacterTextSplitter

def split_text(data,query):
    text_splitter=CharacterTextSplitter(separator="\n",chunk_size=3000,chunk_overlap=200,length_function=len)
    text=text_splitter.split_documents(data)
    
    llm=VertexAI(max_output_tokens=1024,max_retries=6,temperature=0.2)
    template ='''
        {text}
        
        You are a world class journalist, and you will try to summarize the text above in order to create a twitter thread about {query}
        
        Please follow all of the following rules:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content is not too tong, it should be no more than 5—7 tweets
        3/ The content should address the {query} topic very welt
        4/ The content needs to be viral, and get at least 1000 likes
        5/ The content needs to be written in a way that is easy to read and understand
        6/ The content needs to give audience actionable advice & insights too.
        
        SUMMARY:
    '''
    
    prompt_template=PromptTemplate(input_variables=["text","query"],template=template)
    summarization_chain=LLMChain(llm=llm,prompt=prompt_template,verbose=True)
    
    summaries=[]
    
    for chunk in enumerate(text):
        summary=summarization_chain.predict(text=chunk, query=query)
        summaries.append(summary)
    
    return summaries

summaries=split_text(content,query)
st.write(summaries)