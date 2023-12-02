##########################################################################################
# Description: Simple GUI app code based on GradIO demonstrating the pipeline.
##########################################################################################

import os
import random
import re
import shutil
import sys

sys.path.append("../")

import gradio as gr
import openai
import radqg.configs as configs
from radqg.generator import Generator
from radqg.llm.openai import embed_fn as openai_embed_fn
from radqg.llm.openai import qa as openai_qa
from radqg.parse_html import retrieve_figures, retrieve_articles

# ----------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
# remove vector_db


def remove_vector_db():
    for folder in os.listdir(configs.VECTOR_DB_DIR):
        if os.path.isdir(os.path.join(configs.VECTOR_DB_DIR, folder)):
            shutil.rmtree(
                os.path.join(configs.VECTOR_DB_DIR, folder), ignore_errors=True
            )


# ----------------------------------------------------------------------------------------
# generate_question


def generate_question(question_type: str):
    global generator, article_names, figpaths, captions, sampler

    if generator is None:
        # Selecting three articles
        articles_to_include_full_names = [
            "CT Findings of Acute Small-Bowel Entities _ RadioGraphics.html",
            "Imaging of Drug-induced Complications in the Gastrointestinal System _ RadioGraphics.html",
            "Internal Hernias in the Era of Multidetector CT_ Correlation of Imaging and Surgical Findings _ RadioGraphics.html",
        ]

        # Setting up the generator
        generator = Generator(
            data_dir=configs.TOY_DATA_DIR,
            embed_fn=openai_embed_fn,
            selected_articles=articles_to_include_full_names,
        )

        # Setting up the question bank
        article_names, figpaths, captions, sampler = generator.setup_qbank()

    while True:
        # Selecting a figure
        article_name, figpath, caption = generator.select_figure(
            article_names, figpaths, captions, sampler, reset_memory=False
        )

        # Generating a question

        if question_type == "Random":
            question_type = random.choice(["MCQ", "Short-Answer", "Long-Answer"])
        elif question_type == "Multiple choice":
            question_type = "MCQ"
        elif question_type == "Short answer (suitable for flash cards)":
            question_type = "Short-Answer"
        elif question_type == "Open-ended (suitable for essay exams)":
            question_type = "Long-Answer"

        try:
            qa_dict, *_ = generator.generate_qa(
                qa_fn=openai_qa,
                article_name=article_name,
                figpath=figpath,
                caption=caption,
                type_of_question=question_type,
                complete_return=True,
            )
            break
        except AssertionError:
            print("AssertionError occured; trying again...")
            continue

    question = qa_dict["question"]
    if "options" in qa_dict.keys():
        question += "\n\n" + re.sub(r"(, )?([B-E]\))", r"\n\2", qa_dict["options"])
    answer = f'{qa_dict["answer"]}\n\nSource: {article_name.split(" _ RadioGraphics.html")[0]}'

    return [figpath, question, answer]


# ----------------------------------------------------------------------------------------
# GradIO App
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# run_gui


def run_gui():
    remove_vector_db()

    global generator, article_names, figpaths, captions, sampler
    generator = None

    try:
        gr.close_all()
    except:
        pass

    css_style = """
    #normal {font-weight: normal; font-size: 16px}
    #central {font-weight: normal; text-align: center; font-size: 16px}
    #button {background: #5adbb5; font-size: 18px}
    #accordion {font-weight: bold; font-size: 18px; background-color: AliceBlue}
    #articles {background: MintCream; font-size: 16px}
    #title {
        text-align: center;
        width: 90%;
        margin: auto !important;
        font-style: italic;
    }
    #link_out, #authors_list, #authors_affiliation {
        text-align: center;
        min-height: none !important;
    }
    """

    with gr.Blocks(css=css_style) as app:
        gr.Markdown(
            "# RadQG: Radiology Question Generator with Large Language Models",
            elem_id="title",
        )
        gr.HTML(
            "Pouria Rouzrokh, MD, MPH, MHPE<sup>1,2</sup>, Mark M. Hammer, MD<sup>1</sup>, Christine (Cooky) O. Menias, MD<sup>1</sup>",
            elem_id="authors_list",
        )
        gr.HTML(
            """1. RadioGraphics Trainee Editorial Board (RG Team)<br>\
            2. Mayo Clinic Artificial Intelligence Lab, Department of Radiology,\
                Mayo Clinic""",
            elem_id="authors_affiliation",
        )
        gr.Markdown(
            "[[GitHub](https://github.com/PouriaRouzrokh/RadQG)]", elem_id="central"
        )

        with gr.TabItem("Demo"):
            gr.Markdown(
                """
                    Welcome! RadQG is an artificial intelligence (AI) tool that uses large language models to generate knowledge test radiology questions from articles published in the [RadioGraphics](https://pubs.rsna.org/journal/radiographics) journal.
                    This demo will allow you to test RadQG capabilities by generating questions from the following RadioGraphics example articles:
                    <ul>
                        <li style="padding-left: 20px;"><a href="https://pubs.rsna.org/doi/full/10.1148/rg.2018170148" target="_blank">CT Findings of Acute Small-Bowel Entities</a></li>
                        <li style="padding-left: 20px;"><a href="https://pubs.rsna.org/doi/full/10.1148/rg.2016150132" target="_blank">Imaging of Drug-induced Complications in the Gastrointestinal System</a></li>
                        <li style="padding-left: 20px;"><a href="https://pubs.rsna.org/doi/full/10.1148/rg.2016150113" target="_blank">Internal Hernias in the Era of Multidetector CT: Correlation of Imaging and Surgical Findings</a></li>
                    </ul>
                """,
                elem_id="normal",
            )

            with gr.Row():
                gr.Markdown(
                    """ 
                    **To start, please select the type of question you desire and press "Generate!"**.
                    > **Note**: Generating questions may take up to three minutes depending on the question type and the availability of the server.
                    """
                )

            with gr.Row():
                question_type = gr.Dropdown(
                    choices=[
                        "Random",
                        "Multiple choice",
                        "Short answer (suitable for flash cards)",
                        "Open-ended (suitable for essay exams)",
                    ],
                    value="Random",
                    label="Question type:",
                    interactive=True,
                    elem_id="normal",
                )
            with gr.Row():
                generate_button = gr.Button("Generate!", elem_id="button")
            with gr.Row():
                image_box = gr.Image(
                    value="../data/fig_placeholder.png",
                    label="Figure",
                    interactive=False,
                    height=300,
                )
            with gr.Row():
                question_box = gr.Textbox(
                    "Currently, no questions are generated.",
                    label="Generated question:",
                    elem_id="normal",
                    max_lines=500,
                )
            with gr.Accordion(
                label="Click to see the answer!", elem_id="accordion", open=False
            ):
                with gr.Row():
                    answer_box = gr.Textbox(
                        "Currently, no answer is generated.",
                        label="Generated Answer",
                        elem_id="normal",
                        lines=5,
                    )

            # Events
            generate_button.click(
                generate_question,
                [question_type],
                [image_box, question_box, answer_box],
            )

    try:
        app.close()
        gr.close_all()
    except:
        pass

    app.queue(
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


# ----------------------------------------------------------------------------------------
# main

if __name__ == "__main__":
    _ = run_gui()
