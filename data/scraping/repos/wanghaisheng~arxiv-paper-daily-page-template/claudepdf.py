# https://huggingface.co/spaces/taesiri/ClaudeReadsArxiv/blob/main/app.py
import os
import re
import tempfile
import os

import arxiv
import gradio as gr
import requests
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from arxiv_latex_extractor import get_paper_content
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi


from coreservice import app


LEADING_PROMPT = "Read the following paper:"

# with open("assets/custom.css", "r", encoding="utf-8") as f:
#     custom_css = f.read()

custom_css = """
div#component-4 #chatbot {
    height: 800px !important;
}

"""


def replace_texttt(text):
    return re.sub(r"\\texttt\{(.*?)\}", r"*\1*", text)


def get_paper_info(paper_id):
    # Create a search query with the arXiv ID
    search = arxiv.Search(id_list=[paper_id])

    # Fetch the paper using its arXiv ID
    paper = next(search.results(), None)

    if paper is not None:
        # Return the paper's title and abstract
        #  remove new lines
        title_ = paper.title.replace("\n", " ").replace("\r", " ")
        summary_ = paper.summary.replace("\n", " ").replace("\r", " ")
        return title_, summary_
    else:
        return None, None


def get_paper_from_huggingface(paper_id):
    try:
        url = f"https://huggingface.co/datasets/taesiri/arxiv_db/raw/main/papers/{paper_id}.tex"
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return None


class ContextualQA:
    def __init__(self, client, model="claude-2.0"):
        self.client = client
        self.model = model
        self.context = ""
        self.questions = []
        self.responses = []

    def load_text(self, text):
        self.context = text

    def ask_question(self, question):
        if self.questions:
            # For the first question-answer pair, don't add HUMAN_PROMPT before the question
            first_pair = f"Question: {self.questions[0]}\n{AI_PROMPT} Answer: {self.responses[0]}"
            # For subsequent questions, include both HUMAN_PROMPT and AI_PROMPT
            subsequent_pairs = "\n".join(
                [
                    f"{HUMAN_PROMPT} Question: {q}\n{AI_PROMPT} Answer: {a}"
                    for q, a in zip(self.questions[1:], self.responses[1:])
                ]
            )
            history_context = f"{first_pair}\n{subsequent_pairs}"
        else:
            history_context = ""

        full_context = f"{self.context}\n\n{history_context}\n"

        prompt = f"{HUMAN_PROMPT}  {full_context} {HUMAN_PROMPT} {question} {AI_PROMPT}"

        response = self.client.completions.create(
            prompt=prompt,
            stop_sequences=[HUMAN_PROMPT],
            max_tokens_to_sample=6000,
            model=self.model,
            stream=False,
        )
        answer = response.completion
        self.questions.append(question)
        self.responses.append(answer)
        return answer

    def clear_context(self):
        self.context = ""
        self.questions = []
        self.responses = []

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = None


def clean_paper_id(raw_id):
    # Remove any leading/trailing spaces
    cleaned_id = raw_id.strip()

    # Extract paper ID from ArXiv URL if present
    match = re.search(r"arxiv\.org\/abs\/([\w\.]+)", cleaned_id)
    if match:
        cleaned_id = match.group(1)
    else:
        # Remove trailing dot if present
        cleaned_id = re.sub(r"\.$", "", cleaned_id)

    return cleaned_id


def load_context(paper_id):
    global LEADING_PROMPT

    # Clean the paper_id to remove spaces or extract ID from URL
    paper_id = clean_paper_id(paper_id)

    # Check if the paper is already on Hugging Face
    latex_source = get_paper_from_huggingface(paper_id)
    paper_downloaded = False

    # If not found on Hugging Face, use arxiv_latex_extractor
    if not latex_source:
        try:
            latex_source = get_paper_content(paper_id)
            paper_downloaded = True
        except Exception as e:
            return None, [(f"Error loading paper with id {paper_id}: {e}",)]

    if paper_downloaded:
        # Save the LaTeX content to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".tex", delete=False
        ) as tmp_file:
            tmp_file.write(latex_source)
            temp_file_path = tmp_file.name

        # Upload the paper to Hugging Face
        try:
            if os.path.getsize(temp_file_path) > 1:
                hf_api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])

                hf_api.upload_file(
                    path_or_fileobj=temp_file_path,
                    path_in_repo=f"papers/{paper_id}.tex",
                    repo_id="taesiri/arxiv_db",
                    repo_type="dataset",
                )
        except Exception as e:
            print(f"Error uploading paper with id {paper_id}: {e}")

    # Initialize the Anthropic client and QA model
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    qa_model = ContextualQA(client, model="claude-2.0")
    context = f"{LEADING_PROMPT}\n{latex_source}"
    qa_model.load_text(context)

    # Get the paper's title and abstract
    title, abstract = get_paper_info(paper_id)
    title = replace_texttt(title)
    abstract = replace_texttt(abstract)

    return (
        qa_model,
        [
            (
                f"Load the paper with id {paper_id}.",
                f"\n**Title**: {title}\n\n**Abstract**: {abstract}\n\nPaper loaded. You can now ask questions.",
            )
        ],
    )


def answer_fn(qa_model, question, chat_history):
    # if question is empty, tell user that they need to ask a question
    if question == "":
        chat_history.append(("No Question Asked", "Please ask a question."))
        return qa_model, chat_history, ""

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    qa_model.client = client

    try:
        answer = qa_model.ask_question(question)
    except Exception as e:
        chat_history.append(("Error Asking Question", str(e)))
        return qa_model, chat_history, ""

    chat_history.append((question, answer))
    return qa_model, chat_history, ""


def clear_context():
    return []


with gr.Blocks(
    theme=gr.themes.Soft(), css=custom_css, title="ArXiv QA with Claude"
) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center; font-size: 24px;'>
        Explore ArXiv Papers in Depth with <code>claude-2.0</code> - Ask Questions and Get Answers Instantly
        </h1>
        """
    )
    # gr.HTML(
    #     """
    # <p style='text-align: justify; font-size: 18px; margin: 10px;'>
    #     Explore the depths of ArXiv papers with our interactive app, powered by the advanced <code>claude-2.0</code> model. Ask detailed questions and get immediate, context-rich answers from academic papers.
    # </p>
    # """
    # )

    gr.HTML(
        """
    <center>
        <a href="https://huggingface.co/spaces/taesiri/ClaudeReadsArxiv?duplicate=true">
            <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space" style="vertical-align: middle; max-width: 100px; margin-right: 10px;">
        </a>
        <span style="font-size: 14px; vertical-align: middle;">
            Duplicate the Space with your Anthropic API Key &nbsp;|&nbsp;
            Follow me on Twitter for more updates: <a href="https://twitter.com/taesiri" target="_blank">@taesiri</a>
        </span>
    </center>
    """
    )

    with gr.Row().style(equal_height=False):
        with gr.Column(scale=2, emem_id="column-flex"):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                avatar_images=("./assets/user.png", "./assets/Claude.png"),
            )

        with gr.Column(scale=1):
            paper_id_input = gr.Textbox(label="Enter Paper ID", value="2310.12103")
            btn_load = gr.Button("Load Paper")
            qa_model = gr.State()

            question_txt = gr.Textbox(
                label="Question", lines=5, placeholder="Type your question here..."
            )

            btn_answer = gr.Button("Answer Question")
            btn_clear = gr.Button("Clear Chat")

    gr.HTML(
        """<center>All the inputs are being sent to Anthropic's Claude endpoints. Please refer to <a href="https://legal.anthropic.com/#privacy">this link</a> for privacy policy.</center>"""
    )

    gr.Markdown(
        "## Acknowledgements\n"
        "This project is made possible through the generous support of "
        "[Anthropic](https://www.anthropic.com/), who provided free access to the `claude-2.0` API."
    )

    btn_load.click(load_context, inputs=[paper_id_input], outputs=[qa_model, chatbot])

    btn_answer.click(
        answer_fn,
        inputs=[qa_model, question_txt, chatbot],
        outputs=[qa_model, chatbot, question_txt],
    )

    question_txt.submit(
        answer_fn,
        inputs=[qa_model, question_txt, chatbot],
        outputs=[qa_model, chatbot, question_txt],
    )

    btn_clear.click(clear_context, outputs=[chatbot])


app.mount("/js", StaticFiles(directory="js"), name="js")
gr.mount_gradio_app(app, demo, path="/")