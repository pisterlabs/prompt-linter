import argparse
import os

import numpy as np
import openai
import torch
import torchvision
import transformers
from loguru import logger
from mmengine.config import Config
from mmselfsup.registry import MODELS
from PIL import Image
from tqdm import tqdm

import clip
from dataset import create_datasets
from default_configs import get_cfg
from train import create_dataloaders, create_model
from utils import convert_weights_to_fp16, process_class_names, seed_everything

N_AUGMENTATIONS = 5  # five-crop

IMG_TEMPLATE = "a photo of a {}."
VID_TEMPLATE = "a video frame of a person {}."

IMAGENET_TEMPLATE = [
    "a photo of a {}.",
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "ImageUCF101": "a photo of a person doing {}.",
    "ImageNet": IMAGENET_TEMPLATE,
    "ImageNetSketch": IMAGENET_TEMPLATE,
    "ImageNetV2": IMAGENET_TEMPLATE,
    "ImageNetA": IMAGENET_TEMPLATE,
    "ImageNetR": IMAGENET_TEMPLATE,
}


def l2_norm(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, p=2, keepdim=True)


def load_selfsup_model(model_type):
    self_sup_configs = {
        "mocov3": "config_files/mmselfsup/mocov3_resnet50_8xb512-amp-coslr-800e_in1k.py",
        "barlowtwins": "config_files/mmselfsup/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py",
        "byol": "config_files/mmselfsup/byol_resnet50_16xb256-coslr-200e_in1k.py",
    }

    paths = {
        "mocov3": "mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth",
        "barlowtwins": "barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth",
        "byol": "byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth",
    }

    config = self_sup_configs[model_type]

    cfg = Config.fromfile(config)

    model = MODELS.build(cfg.model)

    path = paths[model_type]
    path = f"saved_models/{path}"

    logger.info(f"Loading model from {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    mean = state_dict.pop("data_preprocessor.mean")
    std = state_dict.pop("data_preprocessor.std")

    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return {"model": model.cuda().eval(), "std": std, "mean": mean}


@torch.no_grad()
def get_selfsup_visual_features(model, inputs):
    mean = model["mean"] / 255.0
    std = model["std"] / 255.0
    model = model["model"]

    inputs = torchvision.transforms.Normalize(
        mean=mean, std=std)(inputs).cuda()
    features = model([inputs])[0]
    features = torch.nn.AdaptiveAvgPool2d(1)(features).squeeze(-1).squeeze(-1)

    return features.float().cpu()


def get_openai_embeddings(class_names, data_type):
    def request_emb(x):
        return openai.Embedding.create(input=[x], engine="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    class_names = [process_class_names(name) for name in class_names]

    if data_type == "image":
        class_names = [IMG_TEMPLATE.format(name) for name in class_names]

    elif data_type == "video":
        class_names = [
            VID_TEMPLATE.format(name) for name in class_names]

    text_embeddings = []
    for class_name in tqdm(class_names):
        while True:
            try:
                emb = request_emb(class_name)
                break
            except:
                pass
        text_embeddings.append(torch.tensor(emb))

    text_embeddings = torch.stack(text_embeddings)

    return text_embeddings.float().cpu()


@torch.no_grad()
def create_align_model():
    logger.info("Loading ALIGN model .....")

    processor = transformers.AlignProcessor.from_pretrained(
        "kakaobrain/align-base")
    model = transformers.AlignModel.from_pretrained("kakaobrain/align-base")

    return {
        "model": model.cuda().eval(),
        "processor": processor,
    }


@torch.no_grad()
def create_flava_model():
    logger.info("Loading FLAVA model .....")

    model = transformers.FlavaModel.from_pretrained("facebook/flava-full")
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "facebook/flava-full")

    return {
        "model": model.cuda().eval(),
        "tokenizer": tokenizer,
    }


@torch.no_grad()
def create_alt_clip_model():
    logger.info("Loading AltCLIP model .....")

    model = transformers.AltCLIPModel.from_pretrained("BAAI/AltCLIP")
    processor = transformers.AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

    return {
        "model": model.cuda().eval(),
        "tokenizer": processor.tokenizer,
    }


@torch.no_grad()
def get_flava_visual_features(model, inputs):
    model = model["model"]

    img_embeddings = model.get_image_features(pixel_values=inputs.cuda())

    if img_embeddings.ndim == 3:
        img_embeddings = img_embeddings[:, 0, :]

    img_embeddings = l2_norm(img_embeddings.cpu())
    return img_embeddings


@torch.no_grad()
def get_flava_text_features(model, class_names, use_template=False, data_type="image", bz=100):
    class_names = [process_class_names(name) for name in class_names]

    if use_template:
        if data_type == "image":
            class_names = [IMG_TEMPLATE.format(name) for name in class_names]
        elif data_type == "video":
            class_names = [
                VID_TEMPLATE.format(name) for name in class_names]
        else:
            raise ValueError

    tokenizer = model["tokenizer"]
    model = model["model"]

    all_text_embeddings = []
    for i in range(0, len(class_names), bz):
        text_inputs = tokenizer(
            class_names[i: i + bz], padding="max_length", return_tensors="pt"
        )
        text_inputs = {i: j.cuda() for i, j in text_inputs.items()}
        text_embeddings = model.get_text_features(**text_inputs)

        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings[:, 0, :]

        text_embeddings = l2_norm(text_embeddings.cpu())
        all_text_embeddings.append(text_embeddings)

    return torch.cat(all_text_embeddings, dim=0)


@torch.no_grad()
def get_align_visual_features(model, inputs):
    processor = model["processor"]
    model = model["model"]

    dummy_image = Image.new("RGB", (224, 224))
    dummy_inputs = processor(
        text=[" "], images=dummy_image, return_tensors="pt")
    dummy_inputs["pixel_values"] = inputs
    dummy_inputs = {i: j.cuda() for i, j in dummy_inputs.items()}

    outputs = model(**dummy_inputs)

    embeddings = l2_norm(outputs.image_embeds.cpu())
    return embeddings


@torch.no_grad()
def get_align_text_features(model, class_names, use_template=False, data_type="image"):
    class_names = [process_class_names(name) for name in class_names]

    processor = model["processor"]
    model = model["model"]

    if use_template:
        if data_type == "image":
            class_names = [IMG_TEMPLATE.format(name) for name in class_names]
        elif data_type == "video":
            class_names = [
                VID_TEMPLATE.format(name) for name in class_names]
        else:
            raise ValueError

    dummy_image = Image.new("RGB", (224, 224))
    inputs = processor(text=class_names, images=dummy_image,
                       return_tensors="pt")
    inputs = {i: j.cuda() for i, j in inputs.items()}
    outputs = model(**inputs)

    text_embeddings = l2_norm(outputs.text_embeds.cpu())
    return text_embeddings


@torch.no_grad()
def create_clip_model(visual_backbone):
    logger.info("Loading CLIP model .....")

    clip_model, _ = clip.load(visual_backbone, device="cuda")
    clip_model.cuda().eval()
    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    logger.info(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}",
    )
    logger.info(f"Input resolution: {input_resolution}")
    logger.info(f"Context length: {context_length}")
    logger.info(f"Vocab size: {vocab_size}")

    return clip_model


@torch.no_grad()
def create_clip_prompt_model(cfg, train_dataset, eval_dataset, model_checkpoint):
    if isinstance(eval_dataset, dict):
        # In case we have base & new test sets, use new class names
        eval_label_names = list(eval_dataset.values())[-1].label_names
    else:
        eval_label_names = eval_dataset.label_names

    clip_model = create_model(cfg, train_dataset.label_names, eval_label_names)

    clip_model.text_encoder.apply(convert_weights_to_fp16)
    clip_model.image_encoder.apply(convert_weights_to_fp16)
    clip_model.clip_dtype = torch.float16

    logger.info(f"Loading checkpoint from {model_checkpoint}")
    clip_model.load_state_dict(torch.load(model_checkpoint), strict=True)
    clip_model.cuda().eval()

    return clip_model


@torch.no_grad()
def get_clip_text_features(model, class_names, use_template=False,
                           data_type="image", dataset="ImageNet"):
    clip_weights = []
    for classname in class_names:
        classname = classname.replace('_', ' ')

        if use_template:
            if data_type == "image":
                template = CUSTOM_TEMPLATES[dataset]
                template = template if isinstance(
                    template, list) else [template]
                texts = [t.format(classname) for t in template]
            else:
                assert data_type == "video"
                texts = [VID_TEMPLATE.format(classname)]
        else:
            texts = [classname]

        texts = clip.tokenize(texts).cuda()
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        clip_weights.append(class_embedding.float().cpu())

    clip_weights = torch.stack(clip_weights, dim=1)
    return clip_weights.T


@torch.no_grad()
def get_clip_prompt_text_features(model):
    input_embeddings = model.prompter()
    text_embeddings = model.encode_text(input_embeddings)
    text_embeddings = text_embeddings.float().cpu()
    return text_embeddings


@torch.no_grad()
def get_text_features(
    model_type, model, label_names, base_testing, use_template, data_type, dataset
):
    if model_type == "clip":
        return get_clip_text_features(model, label_names, use_template, data_type, dataset)

    elif model_type == "align":
        return get_align_text_features(model, label_names, use_template, data_type)

    elif model_type in ["flava", "alt_clip"]:
        return get_flava_text_features(model, label_names, use_template, data_type)

    elif model_type == "clip_prompt":
        if base_testing:
            model.prompter.train()
        else:
            model.prompter.eval()

        text_embeddings = get_clip_prompt_text_features(model)
        model.eval()

        return text_embeddings

    return get_openai_embeddings(label_names, data_type)


@torch.no_grad()
def get_clip_visual_features(model, inputs):
    inputs = inputs.cuda()

    if inputs.ndim == 5:
        B, C, T, H, W = inputs.shape

        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        inputs = inputs.reshape(-1, C, H, W)

        features = model.encode_image(inputs).float().cpu()

        features = features.reshape(B, T, -1).max(1)[0]
        return features

    features = model.encode_image(inputs).float().cpu()
    return features


@torch.no_grad()
def get_clip_prompt_visual_features(model, inputs):
    inputs = inputs.cuda()
    features = model.encode_image(inputs)
    features = features.float().cpu()
    return features


def crop_duplication(inputs, n_crops):
    if isinstance(inputs, torch.Tensor):
        return inputs.view(-1, 1).repeat(1, n_crops).reshape(-1)

    assert isinstance(inputs, list)
    return [item for sublist in zip(*[inputs]*n_crops) for item in sublist]


@torch.no_grad()
def get_visual_features(
    model_type, model, dataloader, data_type, five_crop, loader_type
):
    features, filenames, labelnames, labels = [], [], [], []

    for batch in tqdm(dataloader):
        input_tensor, batch_labels, batch_labelnames, batch_filenames = batch

        if five_crop and data_type == "video" and loader_type == "train":
            assert input_tensor.ndim == 6
            B, crops, C, T, H, W = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, C, T, H, W)

            batch_labels = crop_duplication(batch_labels, crops)
            batch_labelnames = crop_duplication(batch_labelnames, crops)
            batch_filenames = crop_duplication(batch_filenames, crops)

        elif five_crop and data_type == "image" and loader_type == "train":
            assert input_tensor.ndim == 5
            B, crops, C, H, W = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, C, H, W)

            batch_labels = crop_duplication(batch_labels, crops)
            batch_labelnames = crop_duplication(batch_labelnames, crops)
            batch_filenames = crop_duplication(batch_filenames, crops)

        if model_type == "clip":
            batch_features = get_clip_visual_features(model, input_tensor)

        elif model_type == "clip_prompt":
            batch_features = get_clip_prompt_visual_features(
                model, input_tensor)

        elif model_type == "align":
            batch_features = get_align_visual_features(model, input_tensor)

        elif model_type in ["flava", "alt_clip"]:
            batch_features = get_flava_visual_features(model, input_tensor)

        else:
            batch_features = get_selfsup_visual_features(
                model, input_tensor
            )

        features.append(batch_features)
        labels.append(batch_labels)
        filenames.extend(batch_filenames)
        labelnames.extend(batch_labelnames)

    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        filenames,
        labelnames,
    )


def get_image_save_name(cfg, args):
    save_name = f"{args.model_type}"
    if "clip" in args.model_type:
        save_name += f"-{cfg.MODEL.VIZ_BACKBONE}"
    save_name += f"-{cfg.DATA.DATASET_NAME}"
    if cfg.DATA.N_SHOT >= 1:
        save_name += f"-{cfg.DATA.N_SHOT}nshot-seed{cfg.RNG_SEED}"
    if cfg.DATA.USE_BASE_AND_NEW:
        save_name += "-base-new"
    if cfg.DATA.TARGET_DATASET:
        save_name += f"-target-dataset-{cfg.DATA.TARGET_DATASET}"
    if args.five_crop:
        save_name += "-5crop"
    if args.use_template and "clip" in args.model_type:
        save_name += "-with-template"

    save_name = save_name.replace("/", "")
    return save_name


def get_video_save_name(cfg, args):
    save_name = f"{args.model_type}-{cfg.DATA.DATASET_NAME}"
    if "clip" in args.model_type:
        save_name += f"{cfg.MODEL.VIZ_BACKBONE}"
    save_name += f"-{cfg.DATA.DATASET_NAME}"
    if cfg.DATA.FEWSHOT and not cfg.DATA.USE_ALL_CLASSES:
        save_name += f"-{cfg.DATA.N_SHOT}shot-{cfg.C_WAY}way-seed{cfg.RNG_SEED}"
    elif cfg.DATA.FEWSHOT:
        save_name += f"-{cfg.DATA.N_SHOT}nshot-all-way-seed{cfg.RNG_SEED}"
    elif cfg.DATA.ZEROSHOT:
        save_name += "-zeroshot"

    save_name += f"-train-{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TRAIN_STRIDES[0]}"
    save_name += f"-test-{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_STRIDES[0]}"
    if args.five_crop:
        save_name += "-5crop"
    if args.use_template and "clip" in args.model_type:
        save_name += "-with-template"

    save_name = save_name.replace("/", "")
    return save_name


def get_save_name(cfg, args):
    if cfg.DATA.TYPE == "image":
        return get_image_save_name(cfg, args)
    return get_video_save_name(cfg, args)


def get_config(args):
    cfg = get_cfg(args)

    if args.five_crop:
        cfg.DATALOADER.TRAIN_BATCHSIZE = cfg.DATALOADER.TRAIN_BATCHSIZE // N_AUGMENTATIONS

    if cfg.DATA.TYPE == "video":
        assert len(cfg.DATA.TRAIN_STRIDES) == 1
        assert len(cfg.DATA.TEST_STRIDES) == 1
        cfg.DATA.TRAIN_VIDEO_SAMPLER = "center"
        cfg.DATA.TEST_METHOD = "single_view"

    if args.five_crop:
        cfg.DATA.TRAIN_AUGS = ["resize", "five_crop", "normalize"]
        cfg.DATA.TRAIN_RESIZE = 224
    else:
        cfg.DATA.TRAIN_AUGS = ["resize", "center_crop", "normalize"]
        cfg.DATA.TRAIN_RESIZE = 224

    if args.model_type in ["mocov3", "barlowtwins", "byol"]:
        # remove normalize
        cfg.DATA.TRAIN_AUGS = cfg.DATA.TRAIN_AUGS[:-1]
        cfg.DATA.TEST_AUGS = cfg.DATA.TEST_AUGS[:-1]

    if args.model_type == "align":
        cfg.DATA.TRAIN_RESIZE = 289
        cfg.DATA.TEST_RESIZE = 289
        cfg.DATA.TRAIN_CROP_SIZE = 289
        cfg.DATA.TEST_CROP_SIZE = 289
        cfg.DATA.MEAN = [0.5, 0.5, 0.5]
        cfg.DATA.STD = [0.5, 0.5, 0.5]

    return cfg


def save_features(cfg, args):
    seed_everything(cfg.RNG_SEED)

    train_dataset, eval_dataset = create_datasets(cfg)

    train_dataloader, eval_dataloader = create_dataloaders(
        cfg, train_dataset, eval_dataset
    )

    save_name = get_save_name(cfg, args)
    save_path = os.path.join(args.save_path, save_name)
    logger.info(f"Saving to {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # create model
    logger.info("Creating model...")

    if args.model_type == "clip":
        model = create_clip_model(cfg.MODEL.VIZ_BACKBONE)
    elif args.model_type == "clip_prompt":
        model = create_clip_prompt_model(
            cfg, train_dataset, eval_dataset, args.model_chekpoint
        )
    elif args.model_type == "align":
        model = create_align_model()
    elif args.model_type == "flava":
        model = create_flava_model()
    elif args.model_type == "alt_clip":
        model = create_alt_clip_model()
    else:
        model = load_selfsup_model(args.model_type)

    # eval_dataloader can be [test] only or [test_base, test_new]
    loaders_types = ["train"] + list(eval_dataloader.keys())
    loaders = [train_dataloader] + list(eval_dataloader.values())

    for loader_type, loader in zip(loaders_types, loaders):
        logger.info(f"Saving {loader_type} features...")

        dataset_labelnames = loader.dataset.label_names
        base_testing = all(
            x == y
            for x, y in zip(
                loader.dataset.label_names, train_dataloader.dataset.label_names
            )
        )

        logger.info("-> text features...")
        text_features = get_text_features(
            model_type=args.model_type,
            model=model,
            label_names=dataset_labelnames,
            base_testing=base_testing,
            use_template=args.use_template,
            data_type=cfg.DATA.TYPE,
            dataset=cfg.DATA.DATASET_NAME
        )
        logger.info("-> visual features...")
        visual_features, labels, filenames, labelnames = get_visual_features(
            model_type=args.model_type,
            model=model,
            dataloader=loader,
            data_type=cfg.DATA.TYPE,
            five_crop=args.five_crop,
            loader_type=loader_type
        )

        to_save = [
            text_features,
            visual_features,
            labels,
            filenames,
            labelnames,
            dataset_labelnames,
        ]

        np.savez(f"{save_path}/{loader_type}_features.npz", *to_save)


if __name__ == "__main__":
    logger.info("Saving features...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-chekpoint", type=str, default=None)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["clip", "clip_prompt", "mocov3", "barlowtwins",
                 "byol", "align", "flava", "alt_clip"],
        required=True,
    )
    parser.add_argument("--five-crop", action="store_true")
    parser.add_argument("--use-template", action="store_true")

    arguments = parser.parse_args()
    configs = get_config(arguments)
    save_features(configs, arguments)
