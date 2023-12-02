import copy
import glob
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import gradio as gr
import PIL
from gradio import processing_utils
from gradio_client.client import DEFAULT_TEMP_DIR
from text_generation import Client
from transformers import AutoProcessor
import boto3
import whisper
import base64
# For dino_sam segementation
import copy
import cv2
import torch
import matplotlib.pyplot as plt
import dino_sam_inpainting as D
# Multiclass classification
import multi_class as M
import random

## CoT
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceTextGenInference

# Keyword extraction
from keybert import KeyBERT

kw_model = KeyBERT()

# Dino SAM cfg
config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
grounded_checkpoint = './models/groundingdino_swint_ogc.pth'  # change the path of the model
sam_checkpoint = './models/sam_vit_h_4b8939.pth'
sam_hq_checkpoint = '' #sam_hq_vit_h.pth
use_sam_hq = ''
#    image_path = image_path
#    text_prompt = text_prompt
output_dir = './outputs'
#    box_threshold = box_threshold
#    text_threshold = text_threshold
device = 'cuda'



s3_client = boto3.client('s3')
asr_model = whisper.load_model("medium")

MODELS = [
    "local/idefics-9b-instruct",
]

API_PATHS = {
    "local/idefics-9b-instruct": (
        "http://<idefics_tgi_hostname>:8080"
    ),
}

SYSTEM_PROMPT = [
    """"The following is a conversation between a highly knowledgeable and intelligent visual AI assistant, called RadAide, and a human user, called User. In the following interactions, User and Assistant will converse in natural language, and RadAide will do its best to answer User’s questions. RadAide has the ability to perceive images and reason about the content of visual inputs. It can also process images by following precise instructs. RadAide was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts. The conversation begins:""",
    """\nUser:""",
    "https://miro.medium.com/v2/resize:fit:1332/0*yl2b-bDJeEwKPUI5"
    "Describe the nature of this image.<end_of_utterance>",
    """\RadAide: A tattooed person holding a sign that says, “Teach your children well,” in a crowd of people. In the middle of the sign, there’s an illustration of the earth with 2 raised fists on either side that have a rainbow pride square background, a trans pride circle background, and brown skin tone stripes on the fists. The raised fist is a symbol of solidarity and specifically Black power as popularized by the Black Panther Party in the 1960s. The rainbow pride flag has rainbow stripes and symbolizes general LGBTQ pride. The trans pride flag has pink, blue, and white stripes and celebrates pride for the trans and gender non-conforming umbrella.<end_of_utterance>""",
    "\nUser:How about this image?",
    "https://cdn.pixabay.com/photo/2017/09/25/13/12/puppy-2785074_1280.jpg",
    "Can you describe it too?<end_of_utterance>",
    """\RadAide: A dog is lying on the floor, looking at the camera. It is looking directly at you. The dog has a white body and brown patches on its face and ears. Its eyes are dark. Its nose is black, and it has long, floppy ears, white paws, long fur, big eyes.<end_of_utterance>""",
    "\nUser: What kind of breed is it?<end_of_utterance>",
    """\RadAide: It looks like a Cavalier King Charles Spaniel. These breeds are known for their gentle, affectionate nature. Their long ears and big, brown eyes give them a sweet, expressive face. Cavaliers are small dogs weighing between 13 and 20 pounds. They have a silky, medium-length coat that comes in various colors, including black and tan, tricolor, red and white, or black.<end_of_utterance>""",
    "\nUser: What can you tell me about this breed of dogs?<end_of_utterance>",
    """\RadAide: One specific characteristic of the Cavalier King Charles Spaniel is its friendly and affectionate nature. This breed is known for being extremely sociable and forming strong bonds with their owners. They are often described as "velcro dogs" because they love to be close to their human companions, whether it's sitting on their laps, cuddling on the couch, or simply following them around the house.<end_of_utterance>""",
    "\nUser: ghjkhjabnufs<end_of_utterance>",
    """\RadAide: That doesn’t seem to be a word. Could you ask me another way?<end_of_utterance>""",
    "\nUser: Do you like Cavalier King Charles Spaniel?<end_of_utterance>",
    """\RadAide: I do not have personal opinions as I’m just a computer program. However, cavaliers are known for being excellent family pets due to their gentle and patient demeanor, making them great with children and other pets. Their affectionate nature also makes them ideal therapy dogs, as they can provide comfort and emotional support to those in need.<end_of_utterance>""",
    "\nUser: How many dogs do you see in this image?",
    "https://i.dailymail.co.uk/i/pix/2011/07/01/article-2010308-0CD22A8300000578-496_634x414.jpg",
    "<end_of_utterance>",
    """\nAssistant: There is no dogs in this image. The picture shows a tennis player jumping to volley the ball.<end_of_utterance>""",
]

BAN_TOKENS = (  # For documentation puporse. We are not using this list, it is hardcoded inside `idefics_causal_lm.py` inside TGI.
    "<image>;<fake_token_around_image>"
)
EOS_STRINGS = ["<end_of_utterance>", "\nUser:"]
STOP_SUSPECT_LIST = []

#GRADIO_LINK = "https://huggingfacem4-idefics-playground.hf.space"
GRADIO_LINK = "http://0.0.0.0:7863"
HTTPD_URL = "http://<httpd_uploadserver_hoostname>:8080/"
API_TOKEN = os.getenv("hf_api_token")
DocAid_logo = "example_images/medicine.png"
global orig_image_path

PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics-9b-instruct",
    token=API_TOKEN,
)

BOT_AVATAR = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Monkey patch adapted from gradio.components.image.Image - mostly to make the `save` step optional in `pil_to_temp_file`
def hash_bytes(bytes: bytes):
    sha1 = hashlib.sha1()
    sha1.update(bytes)
    return sha1.hexdigest()


def pil_to_temp_file(img: PIL.Image.Image, dir: str = DEFAULT_TEMP_DIR, format: str = "png") -> str:
    """Save a PIL image into a temp file"""
    bytes_data = processing_utils.encode_pil_to_bytes(img, format)
    temp_dir = Path(dir) / hash_bytes(bytes_data)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filename = str(temp_dir / f"image.{format}")
    if not os.path.exists(filename):
        img.save(filename, pnginfo=processing_utils.get_pil_metadata(img))
    return filename


def add_file(file):
    return file.name, gr.update(label='🖼️ Uploaded!')
    

# Dino SAM
def dino_sam(image_path, text_prompt, text_threshold=0.4, box_threshold=0.5, output_dir='/temp/gradio/outputs'):
    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
    grounded_checkpoint = './models/groundingdino_swint_ogc.pth'  # change the path of the model
    sam_checkpoint = './models/sam_vit_h_4b8939.pth'
    sam_hq_checkpoint = '' #sam_hq_vit_h.pth
    use_sam_hq = ''
    output_dir = '/tmp/gradio/outputs'
    device = 'cuda'

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = D.load_image(image_path)
    # load model
    model = D.load_model(config_file, grounded_checkpoint, device=device)

    rnum = random.randint(10, 100)
    output_file_name = f'{rnum}_{format(os.path.basename(image_path))}'

    # visualize raw image
    image_pil.save(os.path.join(output_dir, output_file_name))

    # run grounding dino model
    boxes_filt, pred_phrases = D.get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = D.SamPredictor(D.build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = D.SamPredictor(D.build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        D.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        D.show_box(box.numpy(), plt.gca(), label)

    #output_file_name = f'{format(os.path.basename(image_path))}'
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f'grounded_sam_{output_file_name}'),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    D.save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
    return f'grounded_sam_{output_file_name}'




# This is a hack to make pre-computing the default examples work.
# During normal inference, we pass images as url to a local file using the method `gradio_link`
# which allows the tgi server to fetch the local image from the frontend server.
# however, we are building the space (and pre-computing is part of building the space), the frontend is not available
# and won't answer. So tgi server will try to fetch an image that is not available yet, which will result in a timeout error
# because tgi will never be able to return the generation.
# To bypass that, we pass instead the images URLs from the spaces repo.
all_images = glob.glob(f"{os.path.dirname(__file__)}/example_images/*")
DEFAULT_IMAGES_TMP_PATH_TO_URL = {}
for im_path in all_images:
    H = gr.Image(im_path, visible=False, type="filepath")
    tmp_filename = H.preprocess(H.value)
    #DEFAULT_IMAGES_TMP_PATH_TO_URL[tmp_filename] = f"https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/{os.path.basename(im_path)}"
    #DEFAULT_IMAGES_TMP_PATH_TO_URL[tmp_filename] = f"/https://bedrock-415275363822.s3.us-east-1.amazonaws.com/uploads/{os.path.basename(im_path)}"
    #print(f"The tem file path {DEFAULT_IMAGES_TMP_PATH_TO_URL[tmp_filename]}")


# Utils to handle the image markdown display logic
def split_str_on_im_markdown(string: str) -> List[str]:
    """
    Extract from a string (typically the user prompt string) the potential images from markdown
    Examples:
    - `User:![](https://favurl.com/chicken_on_money.png)Describe this image.` would become `["User:", "https://favurl.com/chicken_on_money.png", "Describe this image."]`
    - `User:![](/file=/my_temp/chicken_on_money.png)Describe this image.` would become `["User:", "/my_temp/chicken_on_money.png", "Describe this image."]`
    """
    IMAGES_PATTERN = re.compile(r"!\[[^\]]*\]\((.*?)\s*(\"(?:.*[^\"])\")?\s*\)")
    parts = []
    cursor = 0
    for pattern in IMAGES_PATTERN.finditer(string):
        start = pattern.start()
        if start != cursor:
            parts.append(string[cursor:start])
        image_url = pattern.group(1)
        if image_url.startswith("/file="):
            image_url = image_url[6:]  # Remove the 'file=' prefix
        parts.append(image_url)
        cursor = pattern.end()
    if cursor != len(string):
        parts.append(string[cursor:])
    return parts


def is_image(string: str) -> bool:
    """
    There are two ways for images: local image path or url.
    """
    return is_url(string) or string.startswith(DEFAULT_TEMP_DIR)


def is_url(string: str) -> bool:
    """
    Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url
    """
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


def isolate_images_urls(prompt_list: List) -> List:
    """
    Convert a full string prompt to the list format expected by the processor.
    In particular, image urls (as delimited by <fake_token_around_image>) should be their own elements.
    From:
    ```
    [
        "bonjour<fake_token_around_image><image:IMG_URL><fake_token_around_image>hello",
        PIL.Image.Image,
        "Aurevoir",
    ]
    ```
    to:
    ```
    [
        "bonjour",
        IMG_URL,
        "hello",
        PIL.Image.Image,
        "Aurevoir",
    ]
    ```
    """
    linearized_list = []
    for prompt in prompt_list:
        # Prompt can be either a string, or a PIL image
        if isinstance(prompt, PIL.Image.Image):
            linearized_list.append(prompt)
        elif isinstance(prompt, str):
            if "<fake_token_around_image>" not in prompt:
                linearized_list.append(prompt)
            else:
                prompt_splitted = prompt.split("<fake_token_around_image>")
                for ps in prompt_splitted:
                    if ps == "":
                        continue
                    if ps.startswith("<image:"):
                        linearized_list.append(ps[7:-1])
                    else:
                        linearized_list.append(ps)
        else:
            raise TypeError(
                f"Unrecognized type for `prompt`. Got {type(type(prompt))}. Was expecting something in [`str`,"
                " `PIL.Image.Image`]"
            )
    return linearized_list

def cot_langchain_llama27b(query_string: str) -> str:
    inference_server_url_local = "http://infs.cavatar.info:8083"

    llm_local = HuggingFaceTextGenInference(
        inference_server_url=inference_server_url_local,
        max_new_tokens=200,
        top_k=5,
        top_p=0.96,
        typical_p=0.95,
        temperature=0.001,
        repetition_penalty=1.08,
    )
    template = """Use the following pieces of context to fully understand the intent and create sub staks to address the context. Please try not to, 
        make up an answer nor hallucinate. Use five maximum sentences and keep the sub tasks as precise as possible. List all actionable steps in 
        detail. Be cautious to avoid phrasing that might replicate previous inquiries. This will help in obtaining an accurate and detailed answer. 
        Avoid repetition for clarity.

        Question: {question}
        Answer: Understand the intent of the question then break down the {question} in to sub-tasks. """
    
    prompt = PromptTemplate(
        template=template, 
        input_variables= ["question"]
    )

    llm_chain_local = LLMChain(prompt=prompt, llm=llm_local)
    cot_return = llm_chain_local(query_string)["text"].replace("\n", "") 
    return f'. Please follow the sub tasks listed below and organize your answers in a short paragraph with precisise and professional writing style plus duplicate avoidance: {cot_return}'

def fetch_images(url_list: str) -> PIL.Image.Image:
    """Fetching images"""
    return PROCESSOR.image_processor.fetch_images(url_list)


def handle_manual_images_in_user_prompt(user_prompt: str) -> List[str]:
    """
    Handle the case of textually manually inputted images (i.e. the `<fake_token_around_image><image:IMG_URL><fake_token_around_image>`) in the user prompt
    by fetching them, saving them locally and replacing the whole sub-sequence the image local path.
    """
    if "<fake_token_around_image>" in user_prompt:
        splitted_user_prompt = isolate_images_urls([user_prompt])
        resulting_user_prompt = []
        for u_p in splitted_user_prompt:
            if is_url(u_p):
                img = fetch_images([u_p])[0]
                tmp_file = pil_to_temp_file(img)
                resulting_user_prompt.append(tmp_file)
            else:
                resulting_user_prompt.append(u_p)
        return resulting_user_prompt
    else:
        return [user_prompt]


def gradio_link(img_path: str) -> str:
    #url = f"{GRADIO_LINK}/file={img_path}"
    #url = f"{format(os.path.basename(image_path.name))}"
    #url = f"{img_path}"
    #key_name = f'uploads/{os.path.basename(img_path)}'
    new_file_name =str(img_path)[12:]
    #bucket = 'bedrock-415275363822'
    #s3_client.upload_file(Filename=img_path, Bucket=bucket, Key=key_name)
    #url = f"http://radaide.cavatar.info:8080/"
    orig_image_path = img_path
    return f'{HTTPD_URL}{new_file_name}'
    #return "https://{0}.s3.us-east-1.amazonaws.com/{1}".format(bucket, key_name)


def prompt_list_to_markdown(prompt_list: List[str]) -> str:
    """
    Convert a user prompt in the list format (i.e. elements are either a PIL image or a string) into
    the markdown format that is used for the chatbot history and rendering.
    """
    resulting_string = ""
    for elem in prompt_list:
        if is_image(elem):
            if is_url(elem):
                resulting_string += f"![]({elem})"
            else:
                resulting_string += f"![](/file={elem})"
        else:
            resulting_string += elem
    return resulting_string


def prompt_list_to_tgi_input(prompt_list: List[str]) -> str:
    """
    TGI expects a string that contains both text and images in the image markdown format (i.e. the `![]()` ).
    The images links are parsed on TGI side
    """
    result_string_input = ""
    for elem in prompt_list:
        if is_image(elem):
            if is_url(elem):
                result_string_input += f"![]({elem})"
            else:
                result_string_input += f"![]({gradio_link(img_path=elem)})"
        else:
            result_string_input += elem
            
    return result_string_input


def remove_spaces_around_token(text: str) -> str:
    pattern = r"\s*(<fake_token_around_image>)\s*"
    replacement = r"\1"
    result = re.sub(pattern, replacement, text)
    return result


# Chatbot utils
def format_user_prompt_with_im_history_and_system_conditioning(
    current_user_prompt_str: str, current_image: Optional[str], history: List[Tuple[str, str]]
) -> Tuple[List[str], List[str]]:
    """
    Produces the resulting list that needs to go inside the processor.
    It handles the potential image box input, the history and the system conditionning.
    """
    resulting_list = copy.deepcopy(SYSTEM_PROMPT)

    #CoT Alfred
    cot_added_str = cot_langchain_llama27b(current_user_prompt_str.strip()) if ("detail" in current_user_prompt_str.lower() or "elaborate" in current_user_prompt_str.lower() or "comprehen" in current_user_prompt_str.lower() or 'depict' in current_user_prompt_str.lower()) else ""
    
    # Format history
    for turn in history:
        user_utterance, assistant_utterance = turn
        splitted_user_utterance = split_str_on_im_markdown(user_utterance)

        optional_space = ""
        if not is_image(splitted_user_utterance[0]):
            optional_space = " "
        resulting_list.append(f"\nUser:{optional_space}")
        resulting_list.extend(splitted_user_utterance)
        # CoT Alfred
        resulting_list.append(cot_added_str)
        resulting_list.append(f"<end_of_utterance>\nAssistant: {assistant_utterance}")

    # Format current input
    current_user_prompt_str = remove_spaces_around_token(current_user_prompt_str)
    if current_image is None:
        if "![](" in current_user_prompt_str:
            current_user_prompt_list = split_str_on_im_markdown(current_user_prompt_str)
        else:
            current_user_prompt_list = handle_manual_images_in_user_prompt(current_user_prompt_str)

        optional_space = ""
        if not is_image(current_user_prompt_list[0]):
            # Check if the first element is an image (and more precisely a path to an image)
            optional_space = " "
        resulting_list.append(f"\nUser:{optional_space}")
        resulting_list.extend(current_user_prompt_list)
        #CoT Alfred
        resulting_list.append(cot_added_str)
        resulting_list.append("<end_of_utterance>\nAssistant:")
    else:
        # Choosing to put the image first when the image is inputted through the UI, but this is an arbiratrary choice.
        resulting_list.extend(["\nUser:", current_image, f"{current_user_prompt_str}{cot_added_str}<end_of_utterance>\nAssistant:"])
        current_user_prompt_list = [current_user_prompt_str]

    return resulting_list, current_user_prompt_list


# dope_callback = gr.CSVLogger()
# problematic_callback = gr.CSVLogger()

textbox = gr.Textbox(
    placeholder="Upload an image and send a message",
    show_label=False,
    # value="Describe the battle against the fierce dragons.",
    visible=True,
    container=False,
    label="Text input",
    scale=6,
)
with gr.Blocks(title="Multimodal Playground", theme=gr.themes.Base()) as demo:
    gr.HTML("""<h1 align="center">Multimodal Playground</h1>""")
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Image(DocAid_logo, elem_id="banner-image", show_label=False, show_download_button=False, height=200, weight=100)
        with gr.Column(scale=5):
            gr.HTML("""
                <p>📚 The demo presents <strong>Dialogue Guided Visual Language Processing</strong>, a multimodality VLP pirpeline based on LLM (i.e. Llama-v2) and VLM (i.e. IDEFICS) model that processes both image, text and voicenputs.</p>
                <p>🅿️ <strong>Intended uses:</strong> This demo serves as a proof of concept for multimodal generation. To prepare it for production, further refinement, including fine-tuning and expert evaluation, is necessary.</p>
                <p>⛔️ <strong>Limitations:</strong> The model might generate inaccurate information, invent details from images or text, and often overlooks minute image details. Although it generally avoids responding to dubious user queries, it can still produce outputs that may be racist, stereotypical, or offensive, especially when specifically prompted.</p>
            """)

    # with gr.Row():
    #     with gr.Column(scale=2):
    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=MODELS,
            value="local/idefics-9b-instruct",
            interactive=True,
            show_label=False,
            container=False,
            label="Model",
            visible=False,
        )

    imagebox = gr.Image(type="filepath", label="Image input", visible=False)

    with gr.Row():
        # def prefetch_images_in_history(user_prompt_str):
        #     """
        #     Pre-fetch the images that are passed in the chatbot default history.
        #     """
        #     return prompt_list_to_markdown(handle_manual_images_in_user_prompt(user_prompt_str))

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Multimodal",
            visible=True,
            height=1000,
            avatar_images=[None, BOT_AVATAR]
            #                 value=[
            #                     [
            #                         (
            #                             prefetch_images_in_history(
            #                                 """Write a funny story including the following characters.
            # There should be drama and most importantly romance.
            # Character 1:<fake_token_around_image><image:https://m.media-amazon.com/images/M/MV5BODg1OTczMWEtNTU3MS00OTUzLThjODEtNDg1MWQwZmExYmFlXkEyXkFqcGdeQWFybm8@._V1_QL75_UX500_CR0,0,500,281_.jpg><fake_token_around_image>
            # Character 2:<fake_token_around_image><image:https://static.wikia.nocookie.net/dreamworks/images/0/09/Shrek_dreamworks.png/revision/latest?cb=20170921225439><fake_token_around_image>
            # Character 3:<fake_token_around_image><image:https://upload.wikimedia.org/wikipedia/en/e/ef/Marsupilami_1.jpg><fake_token_around_image>
            # The plot should take place in the world pictured here:<fake_token_around_image><image:https://www.astronomy.com/wp-content/uploads/sites/2/2021/09/ON1.jpg><fake_token_around_image>"""
            #                             )
            #                         ),
            #                         """Once upon a time, in a world where the sky was a dome and the ground was covered in grass, there lived a beautiful princess named Mulan. She was kind, brave, and always ready to help others. One day, while walking through the forest, she stumbled upon a strange creature. It was a green ogre named Shrek, who was lost and looking for his way back home.
            # Mulan, being the kind-hearted person she was, decided to help Shrek find his way back. As they traveled together, they began to develop a strong bond. Shrek was impressed by Mulan's bravery and determination, while Mulan admired Shrek's loyalty and sense of humor.
            # However, their journey was not without obstacles. They encountered a mischievous cat named Hoboua, who was always causing trouble and getting into fights. Despite his troublemaking ways, Hoboua had a good heart and eventually became a valuable ally to Mulan and Shrek.
            # As they traveled, they faced many challenges, including battling fierce dragons and navigating treacherous mountains. Through it all, Mulan and Shrek grew closer, and their feelings for each other deepened.
            # Finally, they reached Shrek's home, and he was reunited with his family and friends. Mulan, however, was sad to leave him behind. But Shrek had a surprise for her. He had fallen in love with her and wanted to be with her forever.
            # Mulan was overjoyed, and they shared a passionate kiss. From that day on, they lived happily ever after, exploring the world together and facing any challenges that came their way.
            # And so, the story of Mulan and Shrek's romance came to an end, leaving a lasting impression on all who heard it.""",
            #                     ],
            #                 ],
        )

    with gr.Group():
        with gr.Row():
            with gr.Column():
                textbox.render()
            with gr.Column():
                asr_audio = gr.Audio(
                        label="Input Audio", 
                        show_label=True,
                        source="microphone",
                        type="filepath")
        with gr.Row():
                #textbox.render()
                submit_btn = gr.Button(value="▶️  Submit", visible=True)
                clear_btn = gr.ClearButton([textbox, imagebox, chatbot], value="🧹 Clear")
                regenerate_btn = gr.Button(value="🔄 Regenerate", visible=True)
                upload_btn = gr.UploadButton("📁 Upload image", file_types=["image"])
                asr_btn = gr.Button("😬 Transcribe")
    # with gr.Group():
    #     with gr.Row():
    #         with gr.Column(scale=1, min_width=50):
    #             dope_bttn = gr.Button("Dope🔥")
    #         with gr.Column(scale=1, min_width=50):
    #             problematic_bttn = gr.Button("Problematic😬")

    with gr.Row():
        with gr.Accordion("Advanced settings", open=False, visible=True) as parameter_row:
            max_new_tokens = gr.Slider(
                minimum=8,
                maximum=1024,
                value=512,
                step=1,
                interactive=True,
                label="Maximum number of new tokens to generate",
            )
            repetition_penalty = gr.Slider(
                minimum=0.01,
                maximum=5.0,
                value=1.0,
                step=0.01,
                interactive=True,
                label="Repetition penalty",
                info="1.0 is equivalent to no penalty",
            )
            decoding_strategy = gr.Radio(
                [
                    "Greedy",
                    "Top P Sampling",
                ],
                value="Greedy",
                label="Decoding strategy",
                interactive=True,
                info="Higher values is equivalent to sampling more low-probability tokens.",
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=0.4,
                step=0.1,
                interactive=True,
                visible=False,
                label="Sampling temperature",
                info="Higher values will produce more diverse outputs.",
            )
            decoding_strategy.change(
                fn=lambda selection: gr.Slider.update(
                    visible=(
                        selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
                    )
                ),
                inputs=decoding_strategy,
                outputs=temperature,
            )
            top_p = gr.Slider(
                minimum=0.01,
                maximum=0.99,
                value=0.8,
                step=0.01,
                interactive=True,
                visible=False,
                label="Top P",
                info="Higher values is equivalent to sampling more low-probability tokens.",
            )
            decoding_strategy.change(
                fn=lambda selection: gr.Slider.update(visible=(selection in ["Top P Sampling"])),
                inputs=decoding_strategy,
                outputs=top_p,
            )
            gr.Markdown(
                """<p><strong>💡 Pro tip</strong>:<br>
                You can input an arbitrary number of images at arbitrary positions in the same query.<br>
                You will need to input each image with its URL with the syntax <code>&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL&gt;&lt;fake_token_around_image&gt;</code>.<br>
                For example, for two images, you could input <code>TEXT_1&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_1&gt;&lt;fake_token_around_image&gt;TEXT_2&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_2&gt;&lt;fake_token_around_image&gt;TEXT_3</code>.<br>
                In the particular case where two images are consecutive, it is not necessary to add an additional separator: <code>&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_1&gt;&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_2&gt;&lt;fake_token_around_image&gt;</code>.</p>"""
            )

    def model_inference(
        model_selector,
        user_prompt_str,
        chat_history,
        image,
        decoding_strategy,
        temperature,
        max_new_tokens,
        repetition_penalty,
        top_p,
    ):
        if user_prompt_str.strip() == "" and image is None:
            return "", None, chat_history

        formated_prompt_list, user_prompt_list = format_user_prompt_with_im_history_and_system_conditioning(
            current_user_prompt_str=user_prompt_str.strip(),
            # With CoT
            #current_user_prompt_str=f'{user_prompt_str.strip()}. {cot_langchain_llama27b(user_prompt_str.strip())}',
            current_image=image,
            history=chat_history,
        )

        client_endpoint = API_PATHS[model_selector]
        client = Client(
            base_url=client_endpoint,
            headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
        )

        # Common parameters to all decoding strategies
        # This documentation is useful to read: https://huggingface.co/docs/transformers/main/en/generation_strategies
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "stop_sequences": EOS_STRINGS,
        }

        assert decoding_strategy in [
            "Greedy",
            "Top P Sampling",
        ]
        if decoding_strategy == "Greedy":
            generation_args["do_sample"] = False
        elif decoding_strategy == "Top P Sampling":
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True
            generation_args["top_p"] = top_p
        mask_filename = None
        if image is None:
            chat_history.append([prompt_list_to_markdown(user_prompt_list), ''])
            '''
            words_list = kw_model.extract_keywords(docs=user_prompt_str, keyphrase_ngram_range=(1,3))
            words_list = [*words_list[0],][0].split()
            orig_image_path = re.findall('\((.*?)\)', chat_history[0][0])[0].split('=')[1]
            print(f'{words_list} and with type {type(words_list)}')
            stopwords = ['mask', 'mark', 'edge' 'segment', 'segmentation', 'cut', 'create', 'generate', 'image', 'picture', 'photo']
            top_word = [i for i in words_list if i not in stopwords][0]
            top_n = M.mclass(text_prompt=user_prompt_str, topics=['Others', 'Create image mask', 'Image segmentation'], top_k=1)
            for label, score in top_n:
                print(f'With label: {label} and score: {score}')
                if ('Create image mask' in label or 'Image segmentation' in label) and orig_image_path is not None:
                    filename = dino_sam(image_path=orig_image_path, text_prompt=top_word, output_dir='/temp/gradio/outputs', box_threshold=0.5, text_threshold=0.55)
                    view_mask_filename = f' [View generated image]({HTTPD_URL}outputs/{filename})'
                    mask_filename = f'![](/file=/tmp/gradio/outputs/{filename})'
                    chat_history.append(
                        [
                            f"{prompt_list_to_markdown(user_prompt_list + [view_mask_filename] + [mask_filename])}",
                            '',
                        ]
                    )
                else:
                    # Alfred Case where there is no image OR the image is passed as `<fake_token_around_image><image:IMAGE_URL><fake_token_around_image>`
                    chat_history.append([prompt_list_to_markdown(user_prompt_list), ''])
            '''
        else:
            # Case where the image is passed through the Image Box.
            # Convert the image into base64 for both passing it through the chat history and
            # displaying the image inside the same bubble as the text.
            chat_history.append(
                [
                    f"{prompt_list_to_markdown([image] + user_prompt_list)}",
                    '',
                ]
            )

        query = prompt_list_to_tgi_input(formated_prompt_list)
        print(query)
        #query += cot_langchain_llama27b(user_prompt_str.strip())
        #print(f'New query: {query}')
        stream = client.generate_stream(prompt=query, **generation_args)

        acc_text = ""
        for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                # That's the exit condition
                return

            if text_token in STOP_SUSPECT_LIST:
                acc_text += text_token
                continue

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token.lstrip()

            acc_text += text_token
            last_turn = chat_history.pop(-1)
            last_turn[-1] += acc_text
            if last_turn[-1].endswith("\nUser"):
                # Safeguard: sometimes (rarely), the model won't generate the token `<end_of_utterance>` and will go directly to generating `\nUser:`
                # It will thus stop the generation on `\nUser:`. But when it exits, it will have already generated `\nUser`
                # This post-processing ensures that we don't have an additional `\nUser` wandering around.
                last_turn[-1] = last_turn[-1][:-5]
            chat_history.append(last_turn)
            yield "", None, chat_history
            acc_text = ""

    def asr_inference(audio):
        audio = whisper.load_audio(audio)
        audio = whisper.pad_or_trim(audio)
    
        mel = whisper.log_mel_spectrogram(audio).to(asr_model.device)
    
        _, probs = asr_model.detect_language(mel)
    
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(asr_model, mel, options)
    
        return(result.text)
    
    def model_inference_asr(
        model_selector,
        audio,
        chat_history,
        image,
        decoding_strategy,
        temperature,
        max_new_tokens,
        repetition_penalty,
        top_p,
    ):
        user_prompt_str = asr_inference(audio)
        acc_text = ""
        if user_prompt_str.strip() == "" and image is None:
            return "", None, chat_history

        formated_prompt_list, user_prompt_list = format_user_prompt_with_im_history_and_system_conditioning(
            current_user_prompt_str=user_prompt_str.strip(),
            current_image=image,
            history=chat_history,
        )

        client_endpoint = API_PATHS[model_selector]
        client = Client(
            base_url=client_endpoint,
            headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
        )

        # Common parameters to all decoding strategies
        # This documentation is useful to read: https://huggingface.co/docs/transformers/main/en/generation_strategies
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "stop_sequences": EOS_STRINGS,
        }

        print(f'Chat_history:{type(chat_history)} and the 1st {chat_history[0]}')
        orig_image_path = re.findall('\((.*?)\)', chat_history[0][0])[0].split('=')[1]
        print(f'...... and the image_path {orig_image_path}')

        assert decoding_strategy in [
            "Greedy",
            "Top P Sampling",
        ]
        if decoding_strategy == "Greedy":
            generation_args["do_sample"] = False
        elif decoding_strategy == "Top P Sampling":
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True
            generation_args["top_p"] = top_p
        mask_filename = None
        if image is None:
            words_list = kw_model.extract_keywords(docs=user_prompt_str, keyphrase_ngram_range=(1,3))
            words_list = [*words_list[0],][0].split()
            print(f'{words_list} and with type {type(words_list)}')
            stopwords = ['mask', 'create', 'generate', 'image', 'cut', 'edge', 'picture', 'photo']
            top_word = [i for i in words_list if i not in stopwords][0]
            if ("mask" in user_prompt_str.lower() or "segment" in user_prompt_str.lower()) and orig_image_path is not None:
                print(f'Here {orig_image_path} with mask prompt {top_word} !')
                filename = dino_sam(image_path=orig_image_path, text_prompt=top_word, output_dir='/temp/gradio/outputs', box_threshold=0.5, text_threshold=0.55)
                view_mask_filename = f' [View generated image]({HTTPD_URL}outputs/{filename})'
                mask_filename = f'![](/file=/tmp/gradio/outputs/{filename})'
                chat_history.append(
                    [
                        #f"{prompt_list_to_markdown(user_prompt_list + ['[ -> Generated image](http://radaide.cavatar.info:8080/outputs/mask_filename)'])}",
                        #OK f"{prompt_list_to_markdown(user_prompt_list + [mask_filename])}",
                        f"{prompt_list_to_markdown(user_prompt_list + [view_mask_filename] + [mask_filename])}",
                        #f"{'![](/file=/temp/gradio/outputs/{filename})'+ prompt_list_to_markdown(user_prompt_list)}",
                        '',
                    ]
                )
            else:
               chat_history.append([prompt_list_to_markdown(user_prompt_list), ''])
        else:
            # Case where the image is passed through the Image Box.
            # Convert the image into base64 for both passing it through the chat history and
            # displaying the image inside the same bubble as the text.
            chat_history.append(
                [
                    f"{prompt_list_to_markdown([image] + user_prompt_list)}",
                    '',
                ]
            )

        query = prompt_list_to_tgi_input(formated_prompt_list)
        stream = client.generate_stream(prompt=query, **generation_args)

        if mask_filename is not None:
            yield "", None, chat_history
        else:
            for idx, response in enumerate(stream):
                text_token = response.token.text

                if response.details:
                    # That's the exit condition
                    return

                if text_token in STOP_SUSPECT_LIST:
                    acc_text += text_token
                    continue

                if idx == 0 and text_token.startswith(" "):
                    text_token = text_token.lstrip()

                acc_text += text_token
                last_turn = chat_history.pop(-1)
                last_turn[-1] += acc_text
                if last_turn[-1].endswith("\nUser"):
                    # Safeguard: sometimes (rarely), the model won't generate the token `<end_of_utterance>` and will go directly to generating `\nUser:`
                    # It will thus stop the generation on `\nUser:`. But when it exits, it will have already generated `\nUser`
                    # This post-processing ensures that we don't have an additional `\nUser` wandering around.
                    last_turn[-1] = last_turn[-1][:-5]
                chat_history.append(last_turn)
                yield "", None, chat_history
                acc_text = ""

    def process_example(message, image):
        """
        Same as `model_inference` but in greedy mode and with the 80b-instruct.
        Specifically for pre-computing the default examples.
        """
        model_selector="local/idefics-9b-instruct"
        user_prompt_str=message
        chat_history=[]
        max_new_tokens=512

        formated_prompt_list, user_prompt_list = format_user_prompt_with_im_history_and_system_conditioning(
            current_user_prompt_str=user_prompt_str.strip(),
            current_image=image,
            history=chat_history,
        )

        client_endpoint = API_PATHS[model_selector]
        client = Client(
            base_url=client_endpoint,
            headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
            timeout=240, # Generous time out just in case because we are in greedy. All examples should be computed in less than 30secs with the 80b-instruct.
        )

        # Common parameters to all decoding strategies
        # This documentation is useful to read: https://huggingface.co/docs/transformers/main/en/generation_strategies
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": None,
            "stop_sequences": EOS_STRINGS,
            "do_sample": False,
        }

        if image is None:
            # Case where there is no image OR the image is passed as `<fake_token_around_image><image:IMAGE_URL><fake_token_around_image>`
            chat_history.append([prompt_list_to_markdown(user_prompt_list), ''])
        else:
            # Case where the image is passed through the Image Box.
            # Convert the image into base64 for both passing it through the chat history and
            # displaying the image inside the same bubble as the text.
            chat_history.append(
                [
                    f"{prompt_list_to_markdown([image] + user_prompt_list)}",
                    '',
                ]
            )

        # Hack - see explanation in `DEFAULT_IMAGES_TMP_PATH_TO_URL`
        for idx, i in enumerate(formated_prompt_list):
            if i.startswith(DEFAULT_TEMP_DIR):
                for k, v in DEFAULT_IMAGES_TMP_PATH_TO_URL.items():
                    if k == i:
                        formated_prompt_list[idx] = v
                        break

        query = prompt_list_to_tgi_input(formated_prompt_list)
        print(query)
        generated_text = client.generate(prompt=query, **generation_args).generated_text
        if generated_text.endswith("\nUser"):
            generated_text = generated_text[:-5]

        last_turn = chat_history.pop(-1)
        last_turn[-1] += generated_text
        chat_history.append(last_turn)
        return "", None, chat_history

    textbox.submit(
        fn=model_inference,
        inputs=[
            model_selector,
            textbox,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[textbox, imagebox, chatbot],
    )
    submit_btn.click(
        fn=model_inference,
        inputs=[
            model_selector,
            textbox,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[
            textbox,
            imagebox,
            chatbot,
        ],
    )

    def remove_last_turn(chat_history):
        if len(chat_history) == 0:
            return gr.Update(), gr.Update()
        last_interaction = chat_history[-1]
        chat_history = chat_history[:-1]
        chat_update = gr.update(value=chat_history)
        text_update = gr.update(value=last_interaction[0])
        return chat_update, text_update

    regenerate_btn.click(fn=remove_last_turn, inputs=chatbot, outputs=[chatbot, textbox]).then(
        fn=model_inference,
        inputs=[
            model_selector,
            textbox,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[
            textbox,
            imagebox,
            chatbot,
        ],
    )
    asr_btn.click(
        fn=model_inference_asr,
        inputs=[
            model_selector,
            asr_audio,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[
            textbox,
            imagebox,
            chatbot,
        ],
    )
    upload_btn.upload(add_file, [upload_btn], [imagebox, upload_btn], queue=False)
    submit_btn.click(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    textbox.submit(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    clear_btn.click(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    asr_btn.click(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    
    examples_path = os.getcwd()
    gr.Examples(
        examples=[
            [
                (
                    "Which device produced this image? Please explain the main clinical purpose of such image?"
                    "Can you write a radiology report based on this image?"
                ),
                f"{examples_path}/example_images/chest-ct.jpg",
            ],
            [
                "Can you describe the nature of this image? Do you think it's real?",
                f"{examples_path}/example_images/fashion_12.jpg",
            ],
            [
                "Can you describe the action on this image? How many animals total are there in this image? Please identify the species by name with best effort.",
                f"{examples_path}/example_images/assets/demo8.jpg",
            ],
            [
                "Name the sport from this image? Please identify the player's role by name with best effort.",
                f"{examples_path}/example_images/college_football.jpg",
            ],
        ],
        inputs=[textbox, imagebox],
        outputs=[textbox, imagebox, chatbot],
        fn=process_example,
        cache_examples=True,
        examples_per_page=6,
        label=(
            "Click on any example below to get started.\nFor convenience, the model generations have been"
            " pre-computed with `idefics-9b-instruct`."
        ),
    )


demo.queue(concurrency_count=40, max_size=40)
demo.launch(debug=True, server_name="0.0.0.0", server_port=7863, height=2048, share=False, ssl_verify=False, ssl_keyfile="<ssl_cert.key>", ssl_certfile="<ssl_cert.pem>", auth=("<username>", "<>passcode"))
