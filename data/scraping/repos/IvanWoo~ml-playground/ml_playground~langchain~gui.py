import panel as pn
import param

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from ml_playground.langchain.utils.db import from_documents


def load_db(model="gpt-3.5-turbo", chain_type="stuff", k=4):
    embeddings = OpenAIEmbeddings()

    vectordb = from_documents(
        [],
        embeddings,
    )
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    llm = ChatOpenAI(model_name=model, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.qa = load_db()

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(
                pn.Row("User:", pn.pane.Markdown("", width=600)), scroll=True
            )
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result["answer"]
        self.panels.extend(
            [
                pn.Row("User:", pn.pane.Markdown(query, width=600)),
                pn.Row(
                    "ChatBot:",
                    pn.pane.Markdown(
                        self.answer, width=600, style={"background-color": "#F6F6F6"}
                    ),
                ),
            ]
        )
        inp.value = ""  # clears loading indicator when cleared
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends(
        "db_query",
    )
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(
                    pn.pane.Markdown(
                        f"Last question to DB:", styles={"background-color": "#F6F6F6"}
                    )
                ),
                pn.Row(pn.pane.Str("no DB accesses so far")),
            )
        return pn.Column(
            pn.Row(
                pn.pane.Markdown(f"DB query:", styles={"background-color": "#F6F6F6"})
            ),
            pn.pane.Str(self.db_query),
        )

    @param.depends(
        "db_response",
    )
    def get_sources(self):
        if not self.db_response:
            return
        rlist = [
            pn.Row(
                pn.pane.Markdown(
                    f"Result of DB lookup:", styles={"background-color": "#F6F6F6"}
                )
            )
        ]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends("convchain", "clr_history")
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(
                pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True
            )
        rlist = [
            pn.Row(
                pn.pane.Markdown(
                    f"Current Chat History variable",
                    styles={"background-color": "#F6F6F6"},
                )
            )
        ]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self, count=0):
        self.chat_history = []
        return


pn.extension()
cb = cbfs()

button_clearhistory = pn.widgets.Button(name="Clear History", button_type="warning")
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder="Enter text here…")

conversation = pn.bind(cb.convchain, inp)

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2 = pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
)
tab3 = pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(
        button_clearhistory,
        pn.pane.Markdown("Clears chat history. Can use to start a new topic"),
    ),
    pn.layout.Divider(),
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown("# Chat With Kolena Repo")),
    pn.Tabs(
        ("Conversation", tab1),
        ("Database", tab2),
        ("Chat History", tab3),
        ("Configure", tab4),
    ),
)
dashboard.servable()
