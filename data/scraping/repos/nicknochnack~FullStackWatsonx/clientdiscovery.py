import reflex as rx
import os

# Import app components
from app.components.navbar import navbar
from app.components.chat import chat, action_bar
from app.components.layout import info_box

# Load overall application state
from app.config.state import State
from app.config import style

# Import Langchain interface  and use base chain
from app.utils.watsonxga import WatsonxLangchainLLM
from app.utils.watsonxworkbench import WatsonxWorkbenchLangchainLLM
from app.utils.vectorstore import VectorStore
from app.utils.prompts import ragprompt

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

os.environ["TOKENIZERS_PARALLELISM"] = "False"

config = rx.config.get_config()

if config.watsonx_type == "ga":
    # Create Starcoder chain
    llm = WatsonxLangchainLLM(
        model_id=ModelTypes.LLAMA_2_70B_CHAT.value,
        generate_params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 200,
        },
    ).from_pretrained()

if config.watsonx_type == "workbench":
    llm = WatsonxWorkbenchLangchainLLM(
        model_id="meta-llama/llama-2-13b-chat",
        generate_params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 200,
            "stream": True,
        },
    ).from_pretrained()

store = None


class ClientDiscoveryState(State):
    """The page state."""

    question: str = ""
    chat_history: list[tuple[str, str]]

    store_created: bool = False
    file_name: str
    background: str
    ideas: str
    is_uploading: bool = False

    async def answer(self):
        if store:
            if config.watsonx_type == "ga":
                documents = store.query(self.question)
                prompt = ragprompt(self.question, documents)

                answer = llm(prompt)
                self.chat_history.append((self.question, answer))
            if config.watsonx_type == "workbench":

                documents = store.query(self.question)
                prompt = ragprompt(self.question, documents)

                answer = ""
                self.chat_history.append((self.question, ""))
                for response in llm.stream(prompt):
                    answer += response
                    self.chat_history[-1] = (self.question, answer)

        else:
            self.chat_history.append((self.question, "Document Not Loaded"))

    async def handle_upload(self, files: list[rx.UploadFile]):
        global store

        for file in files:
            upload_data = await file.read()
            outfile = "assets/profile.pdf"

            # Save the file.
            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)

        store = VectorStore("assets/profile.pdf")
        self.store_created = True

        if config.watsonx_type == "ga":
            question = "Provide a background summary of the background of the person"
            documents = store.query(question)
            prompt = ragprompt(question, documents)
            self.background = llm(prompt)

            question = "What are some potential business ideas that this person would be interested in?"
            documents = store.query(question)
            prompt = ragprompt(question, documents)
            self.ideas = llm(prompt)

        if config.watsonx_type == "workbench":
            question = "Provide a background summary of the background of the person"
            documents = store.query(question)
            prompt = ragprompt(question, documents)

            answer = ""
            for response in llm.stream(prompt):
                answer += response
                self.background = answer

            question = "What are some potential business ideas that this person would be interested in?"
            documents = store.query(question)
            prompt = ragprompt(question, documents)

            answer = ""
            for response in llm.stream(prompt):
                answer += response
                self.ideas = answer

    def run_prompt(self, prompt):
        self.question = prompt

    def clear_state(self):
        self.question = ""
        self.chat_history = []

        self.store_created = False
        self.file_name = ""
        self.background = ""
        self.ideas = ""


color = "rgb(107,99,246)"


def document_upload() -> rx.Component:
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select File",
                    color=color,
                    bg="white",
                    border=f"1px solid {color}",
                ),
                rx.text("Drag and drop files here or click to select files"),
            ),
            border=f"1px dotted {color}",
            padding="5em",
        ),
        rx.hstack(rx.foreach(rx.selected_files, rx.text)),
        rx.button(
            "Upload",
            on_click=[
                lambda: ClientDiscoveryState.set_is_uploading(True),
                lambda: ClientDiscoveryState.handle_upload(rx.upload_files()),
                lambda: ClientDiscoveryState.set_is_uploading(False),
            ],
        ),
        rx.button(
            "Clear",
            on_click=rx.clear_selected_files,
        ),
        padding="5em",
    )


@rx.page(route="/clientdiscovery", title="Client Discovery - watsonx")
def index() -> rx.Component:
    return rx.box(
        navbar(),
        rx.hstack(
            rx.container(
                rx.text(
                    """Upload a client profile on the right by choosing Select File, hit upload to get a background profile and opportunity ideas generated by watsonx.ai and Retrieval Augmented Generation. 
                    Learn more about your client by selecting Who is the client? or What issues concern the client? (then hit ask). Hit Clear when you're done.""",
                    color="#342E5C",
                    font_size="0.9em",
                    font_family=style.SANS,
                    padding_top="1em",
                ),
                rx.hstack(
                    rx.button(
                        "Who is the client?",
                        on_click=lambda: ClientDiscoveryState.run_prompt(
                            "Who is the client from the document?"
                        ),
                        style=style.PROMPT_BUTTON,
                    ),
                    rx.button(
                        "What issues concerns the client?",
                        on_click=lambda: ClientDiscoveryState.run_prompt(
                            "What are some of the issues the client from the document is interested in?"
                        ),
                        style=style.PROMPT_BUTTON,
                    ),
                    rx.button(
                        "Clear",
                        on_click=ClientDiscoveryState.clear_state,
                        style=style.CLEAR_BUTTON,
                    ),
                    style={"margin-top": "10px"},
                    position="sticky",
                    z_index="998",
                    top="0",
                ),
                rx.vstack(
                    chat(ClientDiscoveryState),
                    action_bar(ClientDiscoveryState),
                    height="100vh",
                    display="flex",
                    flex_direction="column",
                ),
                style={"min-width": "450px"},
            ),
            rx.container(
                rx.vstack(
                    rx.cond(
                        ClientDiscoveryState.is_uploading,
                        rx.stack(
                            rx.skeleton(height="20px", speed=1.5),
                            rx.skeleton(height="20px", speed=1.5),
                            rx.skeleton(height="20px", speed=1.5),
                            rx.skeleton(height="20px", speed=1.5),
                            rx.skeleton(height="20px", speed=1.5),
                            width="50%",
                        ),
                        rx.cond(
                            ClientDiscoveryState.store_created,
                            rx.box(
                                info_box(
                                    "📚 Background Information",
                                    ClientDiscoveryState.background,
                                ),
                                info_box(
                                    "💡 Discussion Ideas", ClientDiscoveryState.ideas
                                ),
                            ),
                            document_upload(),
                        ),
                    ),
                    height="100vh",
                    display="flex",
                    flex_direction="column",
                ),
                style={"min-width": "1000px"},
            ),
        ),
    )
