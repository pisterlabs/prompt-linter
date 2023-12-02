from dungeon_driver.mechanics.spell_surge_generator import SpellSurgeGenerator
from dungeon_driver.prompts.prompt_builder import replace_prompt
from dungeon_driver.prompts.img_gen import OPENAI_RPG_SD_AGENT_PROMPT
import gradio as gr
import httpx
from loguru import logger
from dungeon_driver.webui.auth import auth_service

timeout = httpx.Timeout(600.0)


def format_evals(evals):
    result = ""
    for key, value in evals.items():
        result += f"{key}:\n"
        result += f"    Score: {value['score']}\n"
        result += f"    Justification: {value['justification']}\n"
    return result


async def generate_sd_prompt(random_event: str):
    prompt = replace_prompt(OPENAI_RPG_SD_AGENT_PROMPT, {"prompt": random_event})
    response = await make_request(prompt, "prompt")
    logger.info(response.json())
    gen_prompt = response.json()["prompt"]
    evals = response.json()["evaluations"]
    pretty_eval = format_evals(evals)
    return gen_prompt, pretty_eval


async def make_request(prompt, endpoint):
    async with httpx.AsyncClient(timeout=timeout) as client:
        client = auth_service.add_auth_headers(client)
        logger.info(f"Making request to {endpoint}")
        request = {"query": prompt, "session_id": "24"}
        response = await client.post(
            f"http://ai_driver:28001/api/v1/image/{endpoint}",
            json=request,
        )
        logger.info(response.text)
        return response


###########################
def generate_random_event():
    """
    Generates a random event string using SpellSurgeGenerator.

    Returns:
        str: A random event string.
    """
    ssg = SpellSurgeGenerator()
    return ssg.generate()


def create_spell_surge_generator() -> gr.Blocks:
    with gr.Blocks() as spell_surge_gen_tab:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Enter a prompt")
                # temp_1 = gr.Slider(minimum=0, maximum=1.5, label="Temperature 1")
                # temp_2 = gr.Slider(minimum=0, maximum=1.5, label="Temperature 2")

            with gr.Column():
                sd_prompt = gr.Textbox(label="Prompt")
            with gr.Column():
                sd_eval = gr.Textbox(label="Evaluations")

        ssg_btn = gr.Button("Generate Spell Surge")
        ssg_btn.click(fn=generate_random_event, outputs=prompt)
        img_gen_btn = gr.Button("Generate Stable Diffusion Prompt")
        img_gen_btn.click(
            fn=generate_sd_prompt, inputs=prompt, outputs=[sd_prompt, sd_eval]
        )
    return spell_surge_gen_tab
