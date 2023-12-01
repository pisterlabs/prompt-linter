##########################################################################################
# Description: GUI app code based on GradIO demonstrating the pipeline.
##########################################################################################

import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings

import radqg.configs as configs
from radqg.utils.langchain_utils import get_all_chunks
from radqg.utils.langchain_utils import (
    get_vector_db,
    get_retriever,
    retrieval_qa,
)


# ----------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------


def upload_file(files):
    global chain
    global file_paths

    chain = None
    file_paths = [file.name for file in files]
    return file_paths, gr.Textbox.update(
        value=f"{len(file_paths)} files were uploaded."
    )


def initialize_app(openai_api_key, chnuk_size, chunk_overlap, k):
    global chain
    global retriever

    docs = get_all_chunks(
        file_paths, chunk_size=int(chnuk_size), chunk_overlap=int(chunk_overlap)
    )
    embedding_llm = OpenAIEmbeddings(
        model=configs.EMBEDDING_MODEL, openai_api_key=openai_api_key
    )
    vector_db = get_vector_db(
        docs,
        db_name=configs.VECTOR_DB,
        load_from_existing=False,
        embeddings=embedding_llm,
    )
    retriever = get_retriever(
        vector_db,
        search_type=configs.SEARCH_TYPE,
        k=int(k),
        fetch_k=configs.FETCH_K,
        contextual_compressor=configs.COMPRESSOR,
    )
    return gr.Textbox.update("The app is initialized.")


def generate_question(
    model,
    tempreture,
    openai_api_key,
    topic,
    format,
    level_of_difficulty,
    other_characteristics,
):
    chain = retrieval_qa(
        retriever,
        model=model,
        temperature=float(tempreture),
        chain_type=configs.CHAIN_TYPE,
        prompt_keywords={
            "format": format,
            "difficulty_level": level_of_difficulty,
            "criteria": other_characteristics,
        },
        openai_api_key=openai_api_key,
    )

    result = chain({"query": topic})

    return gr.Textbox.update(result["result"])


# ----------------------------------------------------------------------------------------
# GradIO App
# ----------------------------------------------------------------------------------------


def run_gui():
    global chain

    try:
        gr.close_all()
    except:
        pass

    css_style = """
    #accordion {font-weight: bold; font-size: 18px; background-color: AliceBlue}
    #model {background-color: MintCream}
    #central {font-weight: normal; text-align: center; font-size: 16px}
    #normal {font-weight: normal; font-size: 16px}
    #generate_button {background-color: LightSalmon; font-size: 18px}
    #upload_button {background-color: LightSalmon; font-size: 18px}
    #title {
        text-align: center;
        width: 90%;
        margin: auto !important;
        font-style: italic;
    }
    #link_out, #authors_list, #authors_affiliation{
        text-align: center;
        min-height: none !important;
    }
    """

    with gr.Blocks(css=css_style) as app:
        gr.Markdown(
            "# RadQG: Radiology Question Generator with Large Language Models",
            elem_id="title",
        )
        gr.HTML("Pouria Rouzrokh MD MPH MHPE<sup>1,2★</sup>", elem_id="authors_list")
        gr.HTML(
            """1. RadioGraphics Trainee Editorial Board (RG Team)<br>\
            2. Mayo Clinic Artificial Intelligence Lab, Department of Radiology,\
                Mayo Clinic""",
            elem_id="authors_affiliation",
        )

        with gr.TabItem("Upload Source Files"):
            # Elements
            source_files = gr.File()
            with gr.Row():
                gr.Markdown(
                    """Please select the source files that you are going to use \
                    for question generating. The files could be either in the \
                        ".pdf" or ".txt" formats.""",
                    elem_id="central",
                )
            with gr.Row():
                upload_button = gr.UploadButton(
                    "Upload Files",
                    file_types=["file"],
                    file_count="multiple",
                    elem_id="upload_button",
                )
            with gr.Row():
                log_textbox1 = gr.Textbox(
                    "Currently, no source files are uploaded.",
                    label="Upload Status",
                )

            # Events
            upload_button.upload(
                upload_file, upload_button, [source_files, log_textbox1]
            )

        with gr.TabItem("Initialize the App"):
            # Elements
            with gr.Row():
                gr.Markdown(
                    "Please select the model that you want to use:", elem_id="normal"
                )
            with gr.Row():
                model = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"],
                    label="Large Language Model:",
                )
            with gr.Row():
                opanai_api = gr.Markdown("Please enter your OpenAI API key:")
            with gr.Row():
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    value=configs.OPENAI_API_KEY,
                    interactive=True,
                    type="password",
                    elem_id="normal",
                )
            with gr.Row():
                with gr.Accordion(
                    label="Advanced Settings", elem_id="accordion", open=False
                ):
                    with gr.Row():
                        chnuk_size = gr.Number(
                            label="Chunk Size (integer number):",
                            value=configs.CHUNK_SIZE,
                            interactive=True,
                            elem_id="normal",
                        )
                    with gr.Row():
                        chunk_overlap = gr.Number(
                            label="Chunk Overlap (integer number):",
                            value=configs.CHUNK_OVERLAP,
                            interactive=True,
                            elem_id="normal",
                        )
                    with gr.Row():
                        k = gr.Number(
                            label="Nubmer of documents to retrieve (integer number):",
                            value=configs.K,
                            interactive=True,
                            elem_id="normal",
                        )
                    with gr.Row():
                        tempreture = gr.Number(
                            label="LLM tempreture (real number):",
                            value=configs.TEMPERATURE,
                            interactive=True,
                            elem_id="normal",
                        )
            with gr.Row():
                initialize_button = gr.Button(
                    "Initialize the App", elem_id="generate_button"
                )
            with gr.Row():
                log_textbox2 = gr.Textbox(
                    "Currently, the app is not initialized.",
                    label="Initialization Status",
                )

            # Events
            initialize_button.click(
                initialize_app,
                [openai_api_key, chnuk_size, chunk_overlap, k],
                [log_textbox2],
            )

        with gr.TabItem("Generate Questions"):
            # Elements
            with gr.Row():
                gr.Markdown(
                    "Please enter the characteristics of your desired question:",
                    elem_id="normal",
                )
            with gr.Row():
                topic = gr.Textbox(
                    label="Topic of the question:", interactive=True, elem_id="normal"
                )
            with gr.Row():
                format = gr.Textbox(
                    label="Format of the question:", interactive=True, elem_id="normal"
                )
            with gr.Row():
                level_of_difficulty = gr.Textbox(
                    label="Level of difficulty of the question:",
                    interactive=True,
                    elem_id="normal",
                )
            with gr.Row():
                other_characteristics = gr.Textbox(
                    label="Other characteristics of the question:",
                    interactive=True,
                    elem_id="normal",
                )
            with gr.Row():
                generate_button = gr.Button(
                    "Generate Questions", elem_id="generate_button"
                )
            with gr.Row():
                question_box = gr.Textbox(
                    "Currently, no questions are generated.",
                    label="Generated Question",
                    elem_id="normal",
                    max_lines=40,
                )

            # Events
            generate_button.click(
                generate_question,
                [
                    model,
                    tempreture,
                    openai_api_key,
                    topic,
                    format,
                    level_of_difficulty,
                    other_characteristics,
                ],
                question_box,
            )

    try:
        app.close()
        gr.close_all()
    except:
        pass

    app.queue(
        concurrency_count=configs.GR_CONCURRENCY_COUNT,
        status_update_rate="auto",
    )

    out = app.launch(
        # max_threads=4,
        share=configs.GR_PUBLIC_SHARE,
        inline=False,
        show_api=False,
        show_error=True,
        server_port=configs.GR_PORT_NUMBER,
        server_name=configs.GR_SERVER_NAME,
    )
    return out
