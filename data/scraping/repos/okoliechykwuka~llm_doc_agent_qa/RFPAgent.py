from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, BSHTMLLoader, UnstructuredImageLoader
# Import things that are needed generically
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain
#setting a memory for conversations
import panel as pn
import os
from dotenv import load_dotenv
load_dotenv()
memory = ConversationBufferMemory(memory_key="chat_history")
   
    
def qa_agent(file, query, chain_type, k):
    """_summary_

    Args:
        file (_type_): _description_
        query (_type_): _description_
        chain_type (_type_): _description_
        k (_type_): _description_

    Returns:
        _type_: _description_
    """
    llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain(llm=OpenAI(temperature=0))

    # load document
    if file.endswith('pdf'):
        loader = PyPDFLoader(file)
    elif file.endswith('docx'):
        loader = Docx2txtLoader(file)
    elif file.endswith('jpg') or file.endswith('jpg'):
        loader = UnstructuredImageLoader(file, mode="elements")
    else:
        raise ValueError
    
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=3228, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type=chain_type, retriever=retriever)
    
    '--------------------------------- CREATE AGENT ---------------------------------'
    tools = [
    Tool(
        name = "Demo",
        func=qa.run,
        description="use this as the primary source of context information when you are asked the question. \
                    Always search for the answers using only the provided tool, don't make up answers yourself"
        
    ),
    
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for answering math-related questions within the given document. Avoid speculating beyond the document's content. If you don't know the answer to a question, simply state 'I don't know'.",
       return_direct=True #return tool directly to the user
    )

    ]
    # Construct the agent. We will use the default agent type here.
    # See documentation for a full list of options.

    agent = initialize_agent(
        tools,
        agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        memory=memory,
        verbose=True,
        )
    
    result = agent.run(input = query)

    return result

#'Explain what the proposed Approach in this Paper is all about'

'------------------------------ Panel App ---------------------------------'

pn.extension('texteditor', template="bootstrap", sizing_mode='stretch_width',theme='dark' )
pn.state.template.param.update(
    main_max_width="690px",
    header_background="blue",
    title='DocumentAgent Application'
)

#######Widget###########
file_input = pn.widgets.FileInput(width=300)
openaikey = pn.widgets.PasswordInput(
    value="", placeholder="Enter your OpenAI API Key here...", width=300
)
prompt = pn.widgets.TextEditor(
    value="", placeholder="Enter your questions here...", height=160, toolbar=False
)
run_button = pn.widgets.Button(name="Run!", margin=(25, 50), background='#f0f0f0', button_type='primary')

select_k = pn.widgets.IntSlider(
    name="Number of relevant chunks", start=1, end=5, step=1, value=2
)
select_chain_type = pn.widgets.RadioButtonGroup(
    name='Chain type', 
    options=['stuff', 'map_reduce', "refine", "map_rerank"],button_type='success'
)


widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        "Chain type:",
        pn.Column(select_chain_type, select_k),
        title="Advanced settings", margin=10
    ), width=600
)

convos = []  # store all panel objects in a list

def agent_app(_):
    os.environ["OPENAI_API_KEY"] = openaikey.value
    
    # save pdf file to a temp file 
    if file_input.value is not None:
        file_input.save(f"/.cache/{file_input.filename}")
    
        prompt_text = prompt.value
        if prompt_text:
            result = qa_agent(file=f"/.cache/{file_input.filename}", query=prompt_text, chain_type=select_chain_type.value, k=select_k.value)
            convos.extend([
                pn.Row(
                    pn.panel("\U0001F60A", width=10),
                    prompt_text,
                    width=600
                ),
                pn.Row(
                    pn.panel("\U0001F916", width=10),
                    pn.Column(
                        "Relevant source text:",
                        pn.pane.Markdown(result)
                    )
                )
            ])
            #return convos
    return pn.Column(*convos, margin=15, width=575, min_height=400)


qa_interactive = pn.panel(
    pn.bind(agent_app, run_button),
    loading_indicator=True,
)

output = pn.WidgetBox('*Output will show up here:*', qa_interactive, width=630, scroll=True)
# Apply CSS styles to the WidgetBox
output.background = 'blue'
# layout
pn.Column(
    pn.pane.Markdown("""
    ## \U0001F60A! Question Answering Agent with your Document file
    
    1) Upload a Document in [pdf, docx, .jpg, html] format. 2) Enter OpenAI API key. This costs $. Set up billing at [OpenAI](https://platform.openai.com/account). 3) Type a question and click "Run".
    
    """),
    
    pn.Row(file_input,openaikey),
    output,
    widgets,
    css_classes=['body']).servable()






