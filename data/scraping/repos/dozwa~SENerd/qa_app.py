import os 
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import panel as pn

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

pn.extension('texteditor', template="bootstrap", sizing_mode='stretch_width')
pn.state.template.param.update(
    main_max_width="690px",
    header_background="#F08080",
    )

prompt = pn.widgets.TextEditor(
    value="", placeholder="Trage hier deine Frage ein...", height=160, toolbar=False
    )

run_button = pn.widgets.Button(name="Run!")

select_k = pn.widgets.IntSlider(
    name="Number of relevant chunks", start=1, end=5, step=1, value=2
    )

select_chain_type = pn.widgets.RadioButtonGroup(
    name='Chain type', 
    options=['stuff', 'map_reduce', "refine", "map_rerank"],
    value='map_reduce'
    )

#select_temperature = pn.widgets.FloatSlider(
#    name="Temperature", start=0.0, end=1.0, step=0.1, value=0.0
#    )

widgets = pn.Row(
    pn.Column(prompt, run_button, margin=5),
    pn.Card(
        "Chain type:",
        pn.Column(select_chain_type, 
                  select_k, 
                  #select_temperature
                  ),
        title="Advanced settings"
        ), width=670
    )

def qa(query, chain_type, k):
    persist_directory = '/Users/dorianzwanzig/Systems_Engineering/SE-Nerd/Notebook/database/chroma/'
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                     chain_type=chain_type, 
                                     retriever=retriever, 
                                     return_source_documents=True, 
                                     #temperature=temperature
                                     )
    result = qa({"query": query})
    print(result['result'])
    return result

convos = []  # store all panel objects in a list

def qa_result(_):
    prompt_text = prompt.value
    if prompt_text:
        result = qa(query=prompt_text, 
                    chain_type=select_chain_type.value, 
                    k=select_k.value,
                    #temperature=select_temperature.value
                    )
        sources = get_sources(result)
        convos.extend([
            pn.Row(
                pn.panel("\U0001F60A", width=10),
                prompt_text,
                width=600
                ),
            pn.Row(
                pn.panel("\U0001F916", width=10),
                pn.Column(
                    result["result"],
                    "Berücksichtigte Quellen: ",
                    pn.pane.Markdown(''.join(['\n--------------------------------------------------------------------\n', sources]))
                )
            )
            ])
    return pn.Column(*convos, margin=15, width=575, height=400)

def get_sources(result):
    sources = []
    for doc in result["source_documents"]:
        name = doc.metadata['source']
        page = doc.metadata['page']
        source_info = f"Name: {name}, Page: {page}"
        sources.append(source_info)
    return '\n'.join(sources)

qa_interactive = pn.panel(
    pn.bind(qa_result, run_button),
    loading_indicator=True,
)

output = pn.WidgetBox('*Hier wird die Antwort angezeigt:*', qa_interactive, width=670, heigth=400, scroll=True, sizing_mode='fixed')

# layout
pn.Column(
    pn.pane.Markdown("""
    ## \U0001F60A! SE-Nerd QA App \U0001F60A! \n
    
    Step 1: Stelle deine Frage \n
    Step 2: Wähle die Parameter \n
    Step 3: Klicke auf senden \n
    
    """),
    #pn.Row(file_input,OPENAI_API_KEY),
    output,
    widgets
).servable()
