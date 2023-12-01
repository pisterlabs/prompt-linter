from io import BytesIO
import os
import pickle
import uuid
import random

import numpy as np
from dotenv import load_dotenv
import openai
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import timm
from torchvision import transforms
from torchvision.transforms.functional import resize

# Fixed size to limit API usage
SIZE = "256x256"
N_IMAGES = 5

load_dotenv()
token = os.getenv("DALLE_TOKEN")
openai.api_key = token


base_folder = os.path.expanduser("~/data")
embedding_map_file = os.path.join(base_folder, "embedding_map.pickle")


class Encoder(nn.Module):
    # ref: https://www.kaggle.com/code/cbentes/tartu-siamese-network-with-beit-embeddings  
    def __init__(self):
        super().__init__()
        self.backbone_beit = timm.create_model('beit_base_patch16_224_in22k', pretrained=True, num_classes=0)
        self.avgpool1d = nn.AdaptiveAvgPool1d(512)

    def forward(self, x):
        x = resize(x, size=[224, 224])
        x = x / 255.0
        x = x.type(torch.float32)
        outputs = self.backbone_beit(x)
        embedding = self.avgpool1d(outputs)
        return embedding


def get_image_embedding(img):
    img = img.convert('RGB')
    model = Encoder()
    convert_to_tensor = transforms.Compose([transforms.PILToTensor()])
    input_tensor = convert_to_tensor(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = torch.flatten(model(input_batch)[0]).cpu().data.numpy()
    return embedding


def get_embedding_map():
    with open(embedding_map_file, 'rb') as handle:
        return pickle.load(handle)


def write_embedding_map(embedding_map):
    with open(embedding_map_file, 'wb') as handle:
        pickle.dump(embedding_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_dataset_from_name(dataset_name):
    folder = os.path.join(base_folder, dataset_name)
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    return image_files


def load_emoji_dataset():
    """ Open Emoji dataset
    """
    return _load_dataset_from_name("openmoji")


def load_people_dataset():
    """ Open people dataset
    """
    return _load_dataset_from_name("people")


def load_ai_dataset():
    """ AI generated dataset
    """
    return _load_dataset_from_name("ai")


def load_variance_dataset():
    """ Variance generated dataset
    """
    return _load_dataset_from_name("variance")


def load_sample_files_from_dataset(dataset_files, sample_size=6):
    return random.choices(dataset_files, k=sample_size)


def load_sample_images_from_dataset(dataset_files, sample_size=6):
    sample_files = load_sample_files_from_dataset(dataset_files, sample_size=sample_size)
    return [get_image_from_file(f) for f in sample_files]


def _write_images(folder, image_list):
    image_file_list = []
    for img in image_list:
        file_name = os.path.join(folder, f"image_{uuid.uuid4().hex}.png")
        img.save(file_name, format='PNG')
        image_file_list.append(file_name)
    return image_file_list


def write_ai_images(image_list):
    """ Persist AI generated images
    """
    folder = os.path.join(base_folder, "ai")
    if not os.path.exists(folder):
        os.mkdir(folder)
    return _write_images(folder, image_list)


def write_variance_images(image_list):
    """ Persist AI generated variances
    """
    folder = os.path.join(base_folder, "variance")
    if not os.path.exists(folder):
        os.mkdir(folder)
    return _write_images(folder, image_list)


def get_image_from_url(image_url):
    """ Download image from url
    """
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content)).convert('RGBA')


def get_image_from_file(file_path):
    return Image.open(file_path).convert('RGBA')


def variation_image_resize(image, new_size):
    return image.resize(new_size)


def variation_find_similar(image):
    query_vec = get_image_embedding(image)
    embedding_map = get_embedding_map()
    rank = sorted([(k, np.dot(v, query_vec)) for k,v in embedding_map.items()], key=lambda x:x[1], reverse=True)
    top_rank = rank[0:6]
    return [(top_key[0], get_image_from_file(top_key[0])) for top_key in top_rank]


def get_image_from_text(text):
    """ Return a generated image from text
    """
    resp = openai.Image.create(prompt=text, n=N_IMAGES, size=SIZE)
    data = resp['data']
    image_url_list = [x['url'] for x in data]
    return [get_image_from_url(image_url) for image_url in image_url_list]


def _image_as_bytes(image):
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask = image.split()[3])
    img_byte_arr = BytesIO()
    background.convert('RGB').save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def get_image_variations(image):
    image_bytes = _image_as_bytes(image)
    resp = openai.Image.create_variation(image=image_bytes, n=N_IMAGES, size=SIZE)
    data = resp['data']
    image_url_list = [x['url'] for x in data]
    return [get_image_from_url(image_url) for image_url in image_url_list]


def get_non_existent_person(size=(256, 256)):
    person_url = "https://thispersondoesnotexist.com/image"
    img = get_image_from_url(person_url)
    return variation_image_resize(img, new_size=size)


def notebook_show_images(sample_images):
    _, axarr = plt.subplots(1,6, figsize=(12, 10))
    for i in range(6):
        axarr[i].imshow(sample_images[i])
        axarr[i].set_axis_off()
        axarr[i].title.set_text(f"{i}")


def notebook_show_files(sample_files):
    sample_images = [get_image_from_file(f) for f in sample_files]
    _, axarr = plt.subplots(1,6, figsize=(12, 10))
    for i in range(6):
        axarr[i].imshow(sample_images[i])
        axarr[i].set_axis_off()
        axarr[i].title.set_text(f"{i}")


def notebook_show_rank(top_rank):
    top_images = [t[1] for t in top_rank]
    notebook_show_images(top_images)
