import argparse
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import layoutparser as lp
import openai
import pytesseract
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from pydantic import BaseModel, ConfigDict

from create_assistant import create_assistant

load_dotenv()


logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.INFO)
logger = logging.getLogger(__name__)


class Block(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    block: lp.elements.base.BaseLayoutElement
    page_index: int


class CaptionedBlock(Block):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    caption: lp.elements.base.BaseLayoutElement


def get_blocks_and_texts(layouts: list[lp.Layout]) -> tuple[list[Block], list[Block]]:
    blocks = []
    texts = []
    for i, layout in enumerate(layouts):
        for block in layout:
            if block.type in ["Table", "Figure"]:
                # Check if the current block overlaps with any existing block
                for existing_block in blocks:
                    if existing_block.page_index != i:
                        # If the blocks are not on the same page, skip the overlap check
                        continue
                    overlap_area = existing_block.block.intersect(block).area
                    overlap_ratio = overlap_area / block.area
                    if overlap_ratio > 0.5:
                        # If the current block overlaps with an existing block by more than 50%
                        # Check which block is the "superset" block
                        if block.area > existing_block.block.area:
                            # If the current block is larger, replace the existing block with the current block
                            blocks.remove(existing_block)
                            blocks.append(Block(block=block, page_index=i))
                        # If the existing block is larger or equal, skip the current block
                        break
                else:
                    # If the current block does not overlap significantly with any existing block, add it to the list
                    blocks.append(Block(block=block, page_index=i))
            elif block.type == "Text":
                texts.append(Block(block=block, page_index=i))
    return blocks, texts


def caption_blocks(blocks: list[Block], texts: list[Block]) -> list[CaptionedBlock]:
    captioned_blocks = []
    # Find the closest text block to the top and bottom of the figure/table block
    for block in blocks:
        block_bottom_center = (
            (block.block.block.x_1 + block.block.block.x_2) / 2,
            block.block.block.y_2,
        )
        block_top_center = (
            (block.block.block.x_1 + block.block.block.x_2) / 2,
            block.block.block.y_1,
        )
        closest_text = None
        closest_distance = float("inf")
        for text in texts:
            if text.page_index != block.page_index:
                continue
            text_top_center = (
                (text.block.block.x_1 + text.block.block.x_2) / 2,
                text.block.block.y_1,
            )
            text_bottom_center = (
                (text.block.block.x_1 + text.block.block.x_2) / 2,
                text.block.block.y_2,
            )
            distance_to_top = (
                (block_bottom_center[0] - text_top_center[0]) ** 2
                + (block_bottom_center[1] - text_top_center[1]) ** 2
            ) ** 0.5
            distance_to_bottom = (
                (block_top_center[0] - text_bottom_center[0]) ** 2
                + (block_top_center[1] - text_bottom_center[1]) ** 2
            ) ** 0.5
            # Reduce `distance_to_top` by 25% to bias towards picking bottom captions
            distance = min(distance_to_top * 0.75, distance_to_bottom)
            if distance < closest_distance:
                closest_distance = distance
                closest_text = text
        if closest_text is not None:
            captioned_blocks.append(
                CaptionedBlock(
                    block=block.block,
                    caption=closest_text.block,
                    page_index=block.page_index,
                )
            )
    return captioned_blocks


def combine_blocks(captioned_block, pages):
    # Combine block and caption together
    x_1 = min(captioned_block.block.block.x_1, captioned_block.caption.block.x_1)
    y_1 = min(captioned_block.block.block.y_1, captioned_block.caption.block.y_1)
    x_2 = max(captioned_block.block.block.x_2, captioned_block.caption.block.x_2)
    y_2 = max(captioned_block.block.block.y_2, captioned_block.caption.block.y_2)
    return pages[captioned_block.page_index].crop((x_1, y_1, x_2, y_2))


def process_captioned_block(captioned_block, pages, base_path):
    combined_image = combine_blocks(captioned_block, pages)

    # Convert the PIL Image object to base64
    buffered = BytesIO()
    combined_image.save(buffered, format="JPEG")

    # Convert the PIL Image object to a string for caption
    caption_image = pages[captioned_block.page_index].crop(
        (
            captioned_block.caption.block.x_1,
            captioned_block.caption.block.y_1,
            captioned_block.caption.block.x_2,
            captioned_block.caption.block.y_2,
        )
    )
    caption_text = pytesseract.image_to_string(caption_image)

    figures_path = os.path.join(base_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    # Convert the caption text to snake case alpha numeric and truncate, then add .jpg to it
    img_name = re.sub("[^0-9a-zA-Z]+", "_", caption_text)[:30] + ".jpg"
    img_path = os.path.join(figures_path, img_name)

    with open(img_path, "wb") as f:
        f.write(buffered.getvalue())

    return {"image": f"figures/{img_name}", "caption": caption_text}


def process_pdf(content: bytes, model: lp.models.Detectron2LayoutModel, base_path: str):
    pages = convert_from_bytes(content)
    logger.info("PDF converted to images")

    with ThreadPoolExecutor(max_workers=16) as executor:
        layouts = list(executor.map(model.detect, pages))
        logger.info("Layout detection completed")

    blocks, texts = get_blocks_and_texts(layouts)
    logger.info("Blocks and texts extracted")

    captioned_blocks = caption_blocks(blocks, texts)
    logger.info("Captioning completed")

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                lambda captioned_block: process_captioned_block(
                    captioned_block, pages, base_path
                ),
                captioned_blocks,
            )
        )

    return results


def wait_on_run(run, thread, client: openai.OpenAI):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def generate_thread_content(
    pdf_path: str, results: dict, client: openai.OpenAI, assistant_id: str
):
    with open(pdf_path, "rb") as f:
        pdf_file = client.files.create(file=f, purpose="assistants")

    try:
        thread = client.beta.threads.create()

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"{json.dumps(results)}\n\nCreate a thread for this. Your answer must be in JSON, media links should be from the local paths above.",
            file_ids=[pdf_file.id],
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id, assistant_id=assistant_id
        )

        run = wait_on_run(run, thread, client)

        messages = client.beta.threads.messages.list(
            thread_id=thread.id, order="asc", after=message.id
        )

        # TODO: OpenAI can return no new messages somehow (might be a bug, the run completes succesfully but no new messages are listed in the thread), catch this and throw error
        if not messages.data or not messages.data[0].content:
            raise ValueError("Unexpected empty response from OpenAI. Please try again.")

    except Exception as e:
        logger.error(f"Failed to generate thread content: {e}")
        raise
    finally:
        # Delete uploaded PDF file
        try:
            client.files.delete(file_id=pdf_file.id)
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")

    # Extract JSON content from the message
    message_content = messages.data[0].content[0].text.value
    json_content = re.search(r"(```json\n)(.*?)(\n```)", message_content, re.DOTALL)
    if json_content is None:
        json_content = re.search(r"(```\n)(.*?)(\n```)", message_content, re.DOTALL)
    if json_content is not None:
        json_content = json_content.group(2)

    try:
        paper_thread = json.loads(json_content)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(
            "The thread generated by OpenAI was not in the expected JSON format."
        )

    return paper_thread


def process_thread(thread_data, base_path):
    processed_data = []
    media_set = set()
    for data in thread_data:
        cleaned_content = re.sub(
            r"【\d+†source】", "", data["content"]
        )  # Remove all source annotations
        media_list = []
        for media in data.get("media", []):
            if media["path"] and media["path"] not in media_set:
                media_file_path = os.path.join(base_path, media["path"])
                if os.path.isfile(media_file_path):
                    media_list.append(media)
                    media_set.add(media["path"])
        processed_data.append({"content": cleaned_content, "media": media_list})
    return processed_data


def render_markdown(processed_thread):
    markdown_content = ""
    for data in processed_thread:
        markdown_content += data["content"] + "\n"
        for media in data["media"]:
            markdown_content += f'\n<div align="center">\n'
            markdown_content += f'    <img src="{media["path"]}" alt="{media.get("explain", "")}" style="max-width: 75%;">\n'
            markdown_content += "</div>\n"
        markdown_content += "\n---\n\n"
    return markdown_content


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def create_thread(
    pdf_url_or_path: str, output_path: str, client: openai.OpenAI, assistant_id: str
):
    # Extract the PDF name from the URL and remove any file extension at the end
    pdf_name = os.path.splitext(pdf_url_or_path.split("/")[-1])[0]
    base_path = os.path.join(output_path, pdf_name)
    results_path = os.path.join(base_path, "results.json")
    pdf_path = os.path.join(base_path, f"{pdf_name}.pdf")
    thread_path = os.path.join(base_path, "thread.json")
    processed_thread_path = os.path.join(base_path, "processed_thread.json")
    markdown_path = os.path.join(base_path, "processed_thread.md")

    # Check if base path already exists and there is a results.json
    # If so, assume we've run this before and just return results
    if os.path.exists(base_path) and os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        os.makedirs(base_path, exist_ok=True)

        if uri_validator(pdf_url_or_path):
            pdf_content = requests.get(pdf_url_or_path).content
            with open(pdf_path, "wb") as f:
                f.write(pdf_content)
        elif os.path.isfile(pdf_url_or_path):
            shutil.copy(pdf_url_or_path, pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
        else:
            raise ValueError(
                f"Invalid input: {pdf_url_or_path}. It should be a valid URL or a file path."
            )

        model = lp.models.Detectron2LayoutModel(
            config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )

        results = process_pdf(pdf_content, model, base_path)
        # Remove duplicates from results
        results = [dict(t) for t in set(tuple(d.items()) for d in results)]
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    paper_thread = generate_thread_content(pdf_path, results, client, assistant_id)
    with open(thread_path, "w") as f:
        json.dump(paper_thread, f, indent=2)

    # Process the thread
    processed_thread = process_thread(paper_thread, base_path)
    with open(processed_thread_path, "w") as f:
        json.dump(processed_thread, f, indent=2)

    # Save processed thread as a markdown file
    markdown_content = render_markdown(processed_thread)
    with open(markdown_path, "w") as f:
        f.write(markdown_content)

    logger.info(f"Saved all outputs to: {os.path.abspath(base_path)}")

    return base_path


def create_assistant_then_thread(
    pdf_url_or_path: str,
    output_path: str,
    client: openai.OpenAI,
    assistant_kwargs: Optional[dict] = None,
):
    if assistant_kwargs is None:
        assistant_kwargs = {}
    try:
        assistant = create_assistant(client, **assistant_kwargs)
    except Exception:
        logger.error("Failed to create assistant", exc_info=True)
        raise
    try:
        saved_path = create_thread(
            pdf_url_or_path,
            output_path,
            client,
            assistant.id,
        )
    except Exception:
        logger.error("Failed to create thread", exc_info=True)
        raise
    finally:
        try:
            client.beta.assistants.delete(assistant.id)
        except Exception:
            logger.error("Failed to delete assistant", exc_info=True)
            raise
    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a PDF from a URL or a local path."
    )
    parser.add_argument(
        "url_or_path", type=str, help="The URL or local path of the PDF to process."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data",
        help="The output directory to store the results.",
    )
    args = parser.parse_args()

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    create_assistant_then_thread(args.url_or_path, args.output, client)
