# Imports
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple, Union

from dotenv import load_dotenv
import numpy as np
from onnxruntime import InferenceSession
from openai import OpenAI
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    DetrFeatureExtractor,
    DetrForSegmentation,
    pipeline,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)

import question_answer.metadata.pica as metadata

# Loading env variables
load_dotenv()

# Variables
# OpenAI params
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-3.5-turbo-1106"

# Artifact path
artifact_path = Path(__file__).resolve().parent / "artifacts" / "answer"

# PICa formatting/config
img_id = 100  # Random idx for inference
question_id = 1005  # Random idx for inference
# Significant driver of performance with little extra cost
# PICa paper's max = 16, but can set higher if model's speed + context size can handle it
n_shot = 16
coco_path = artifact_path / "coco_annotations"
similarity_path = artifact_path / "coco_clip_new"

# Model setup
transformers_path = artifact_path / "transformers"
onnx_path = artifact_path / "onnx"

# Segmentation model config
tag_model = transformers_path / "facebook" / "detr-resnet-50-panoptic"
max_length = 16
num_beams = 4

# Caption model config
caption_model = transformers_path / "nlpconnect" / "vit-gpt2-image-captioning"

# CLIP Encoders config
clip_processor = transformers_path / "openai" / "clip-vit-base-patch16"
clip_onnx = onnx_path / "clip.onnx"

# Dataset variables
NUM_ORIGINAL_EXAMPLES = metadata.NUM_ORIGINAL_EXAMPLES
NUM_ADDED_EXAMPLES = metadata.NUM_ADDED_EXAMPLES
NUM_TEST_EXAMPLES = metadata.NUM_TEST_EXAMPLES


# Helper/main classes
class PICa_OKVQA:
    """
    Question Answering Class
    """

    def __init__(
        self,
        caption_info: Dict[Any, Any] = None,
        tag_info: Dict[Any, Any] = None,
        questions: Dict[str, List[Dict[str, str]]] = None,
        context_idxs: Dict[str, str] = None,
        question_features: np.ndarray = None,
        image_features: np.ndarray = None,
        evaluate: bool = False,
    ):
        self.evaluate = evaluate
        (
            self.traincontext_caption_dict,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
        ) = self.load_anno(
            "%s/captions_train2014.json" % coco_path,
            "%s/mscoco_train2014_annotations.json" % coco_path,
            "%s/OpenEnded_mscoco_train2014_questions.json" % coco_path,
        )
        (
            self.traincontext_caption_dict,
            _,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
        ) = self.add_anno(
            "%s/admirer-pica.json" % coco_path,
            self.traincontext_caption_dict,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
        )
        if evaluate:
            (
                self.testcontext_caption_dict,
                self.testcontext_tags_dict,
                self.testcontext_answer_dict,
                self.testcontext_question_dict,
            ) = self.add_anno(
                "%s/admirer-pica.json" % coco_path,
                evaluate=evaluate,
            )
            # load cached image representation (Coco caption & Tags)
            self.inputtext_dict = self.load_cachetext(self.testcontext_caption_dict, self.testcontext_tags_dict)
            self.load_similarity(evaluate=evaluate)
            question_dict_keys = list(self.testcontext_question_dict.keys())
            image_ids, question_ids = [key.split("<->")[0] for key in question_dict_keys], [
                key.split("<->")[1] for key in question_dict_keys
            ]
            list_questions = list(self.testcontext_question_dict.values())
            self.questions = {
                "questions": [
                    {"image_id": image_id, "question": question_str, "question_id": quest_id}
                    for image_id, question_str, quest_id in zip(image_ids, list_questions, question_ids)
                ]
            }
        else:
            # load cached image representation (Coco caption & Tags)
            self.inputtext_dict = self.load_cachetext(caption_info, tag_info)
            _ = self.load_similarity(context_idxs, question_features, image_features)
            self.questions = questions

        self.train_keys = list(self.traincontext_answer_dict.keys())

    def answer_gen(self):
        _, _, question_dict = self.load_anno(questions=self.questions)

        if self.evaluate:
            pred_answers = []
            gt_answers = []

        keys = list(question_dict.keys())
        for key in keys:
            img_key = int(key.split("<->")[0])
            question, caption = (
                question_dict[key],
                self.inputtext_dict[img_key],
            )

            context_key_list = self.get_context_keys(
                key,
                n_shot,
            )

            # prompt format following OpenAI QA API
            messages = []
            system_message = {
                "role": "system",
                "content": str(
                    "You are given {n_shot} examples of image content, a question about the image, and an answer. "
                    + "Given a new set of content and question, "
                    + "you are tasked with coming up with an answer in a similar way to the examples. "
                    + "If the content is not enough to answer the question, "
                    + "make up an answer structured as:"
                    + "\n"
                    + "1) an acknowledgment of not knowing the correct answer to the question,"
                    + "\n"
                    + "2) a comedic reply using what you can from the content."
                    + "\n"
                    + "For example, if the question is 'What is the color of the user's shirt?', "
                    + "and the context is 'The user is wearing a shirt with a picture of a cat on it', "
                    + "a good answer could be 'I don't know, but I think the cat is cute!'"
                ),
            }
            messages.append(system_message)
            for ni in range(n_shot):
                if context_key_list is None:
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                else:
                    context_key = context_key_list[ni]
                img_context_key = int(context_key.split("<->")[0])
                while True:  # make sure get context with valid question and answer
                    if (
                        len(self.traincontext_question_dict[context_key]) != 0
                        and len(self.traincontext_answer_dict[context_key][0]) != 0
                    ):
                        break
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                caption = self.traincontext_caption_dict[img_context_key]
                question = self.traincontext_question_dict[context_key]
                answer = self.traincontext_answer_dict[context_key]
                if type(caption) == list:
                    caption = caption[0]  # sometimes annotators messed up
                if type(question) == list:
                    question = question[0]
                if type(answer) == list:
                    answer = answer[0]
                user_message = {
                    "role": "user",
                    "content": str(
                        "Image content: " + caption + "\n" + "Question: " + question + "\n" + "Answer: " + answer
                    ),
                }
                messages.append(user_message)
            current_user_message = {
                "role": "user",
                "content": str("Image content: " + caption + "\n" + "Question: " + question + "\n" + "Answer: "),
            }
            messages.append(current_user_message)
            try:
                response = CLIENT.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
            except Exception as e:
                print(e)
                exit(0)

            pred_answer = response.choices[0].message.content

            if self.evaluate:
                answer = self.testcontext_answer_dict[key]
                pred_answers.append(pred_answer)
                gt_answers.append(answer)
            else:
                return pred_answer

        from question_answer.lit_models.metrics import BertF1Score

        return BertF1Score()(pred_answers, gt_answers)

    def get_context_keys(self, key: str, n: int) -> List[str]:
        """Get context keys based on similarity scores"""
        # combined with Q-similairty (image+question)
        lineid = self.valkey2idx[key]

        # Removing validation key from train similarity arrays if needed
        temp_train_feature = None
        temp_image_train_feature = None
        temp_train_idx = None

        for idx in range(NUM_ORIGINAL_EXAMPLES, NUM_ORIGINAL_EXAMPLES + NUM_ADDED_EXAMPLES):
            question_feature_equal = np.array_equal(self.val_feature[lineid], self.train_feature[idx])
            image_feature_equal = np.array_equal(self.val_feature[lineid], self.image_train_feature[idx])
            if question_feature_equal and image_feature_equal:
                mask = np.ones(len(self.train_feature), dtype=bool)
                mask[[idx]] = False
                temp_train_feature = self.train_feature[mask]
                temp_image_train_feature = self.image_train_feature[mask]
                temp_train_idx = self.train_idx.pop(str(idx))
                break

        removed = temp_train_feature is not None and temp_image_train_feature is not None and temp_train_idx is not None
        if removed:
            question_similarity: np.ndarray = np.matmul(temp_train_feature, self.val_feature[lineid, :])
            # end of Q-similairty
            similarity: np.ndarray = question_similarity + np.matmul(
                temp_image_train_feature, self.image_val_feature[lineid, :]
            )
        else:
            question_similarity: np.ndarray = np.matmul(self.train_feature, self.val_feature[lineid, :])
            # end of Q-similairty
            similarity: np.ndarray = question_similarity + np.matmul(
                self.image_train_feature, self.image_val_feature[lineid, :]
            )

        index: np.ndarray = similarity.argsort()[-n:][::-1]
        return [self.train_idx[str(x)] for x in index]

    def load_similarity(
        self,
        context_idxs: Dict[str, str] = None,
        question_features: np.ndarray = None,
        image_features: np.ndarray = None,
        evaluate=False,
    ):
        # Add question train feature, image train feature, and train idx
        self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % similarity_path)
        self.train_idx: Dict[str, str] = json.load(
            open(
                "%s/okvqa_qa_line2sample_idx_train2014.json" % similarity_path,
                "r",
            )
        )
        self.image_train_feature = np.load(
            "%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy" % similarity_path
        )

        if evaluate:
            context_idxs = dict(list(self.train_idx.items())[NUM_ORIGINAL_EXAMPLES:])
            new_keys = [str(idx) for idx in range(len(context_idxs))]
            context_idxs = dict(zip(new_keys, list(context_idxs.values())))
            self.val_feature = self.train_feature[-NUM_ADDED_EXAMPLES:, :]
            self.image_val_feature = self.image_train_feature[-NUM_ADDED_EXAMPLES:, :]
        else:
            self.val_feature = question_features
            self.image_val_feature = image_features

        val_idx = context_idxs
        self.valkey2idx: Dict[str, int] = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)

    def load_tags(
        self,
        tag_info: Dict[Any, List[str]],
    ) -> Dict[int, str]:
        """Loads tags for an image"""
        tags_dict = {}
        image_ids, list_tags = list(tag_info.keys()), list(tag_info.values())
        # Concatenate tags into one string
        list_str_tags = [tags for tags in list_tags]
        for id in range(len(image_ids)):
            tags_dict[image_ids[id]] = list_str_tags[id]
        return tags_dict

    def load_cachetext(
        self,
        caption_info: Dict[Any, List[str]],
        tag_info: Dict[Any, List[str]],
    ):
        """Loads and adds cachetect to the caption"""
        tags_dict = self.load_tags(tag_info)
        caption_dict = {}
        image_ids, captions = list(caption_info.keys()), list(caption_info.values())
        for id in range(len(image_ids)):
            caption_dict[image_ids[id]] = captions[id] + ". " + list(tags_dict.values())[id]
        return caption_dict

    def load_anno(
        self,
        coco_caption_file: Path = None,
        answer_anno_file: Path = None,
        question_anno_file: Path = None,
        questions: Dict[str, List[Dict[str, str]]] = None,
    ) -> Tuple[Dict[int, List[str]], Dict[str, List[str]], Dict[str, str]]:
        """Loads annotation from a caption file"""
        # Define default dictionaries
        caption_dict: defaultdict[int, List[str]] = defaultdict(list)
        answer_dict: defaultdict[str, List[str]] = defaultdict(list)
        question_dict: defaultdict[str, str] = defaultdict(list)

        # Create caption dictionary
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, "r"))
            if isinstance(coco_caption, dict):
                coco_caption: List[Dict[str, Union[str, int]]] = coco_caption["annotations"]
            for sample in coco_caption:
                caption_dict[sample["image_id"]].append(sample["caption"])  # int -> sample[image_id]

        # Create answer dictionary
        if answer_anno_file is not None:
            answer_data = json.load(open(answer_anno_file, "r"))
            answer_annotations: List[Dict[str, Any]] = answer_data["annotations"]
            for sample in answer_annotations:
                id = str(sample["image_id"]) + "<->" + str(sample["question_id"])
                if id not in answer_dict:
                    answer_dict[id] = [x["answer"] for x in sample["answers"]]

        # Create question dictionary
        if question_anno_file is not None:
            question_data = json.load(open(question_anno_file, "r"))
        else:
            question_data = questions

        question_annotations: List[Dict[str, Union[str, int]]] = question_data["questions"]
        for sample in question_annotations:
            id = str(sample["image_id"]) + "<->" + str(sample["question_id"])
            if id not in question_dict:
                question_dict[id] = sample["question"]

        return dict(caption_dict), dict(answer_dict), dict(question_dict)

    def add_anno(
        self,
        add: Path,
        context_caption_dict: Dict[int, List[str]] = None,
        context_answer_dict: Dict[str, List[str]] = None,
        context_question_dict: Dict[str, str] = None,
        evaluate=False,
    ):
        """Load/add extra annotations to the annotations dictionaries"""
        add_dict = json.load(open(add, "r"))

        context_tag_dict = {}

        caption_add = dict(zip(list(add_dict["image_id"].values()), list(add_dict["caption"].values())))
        tags_add = dict(zip(list(add_dict["image_id"].values()), list(add_dict["tags"].values())))
        combine_ids = [
            str(image_id) + "<->" + str(question_id)
            for image_id, question_id in zip(
                list(add_dict["image_id"].values()), list(add_dict["question_id"].values())
            )
        ]
        answer_add = dict(zip(combine_ids, list(add_dict["answer"].values())))
        question_add = dict(zip(combine_ids, list(add_dict["question"].values())))

        if evaluate:
            context_caption_dict = {}
            context_answer_dict = {}
            context_question_dict = {}
        context_caption_dict.update(caption_add)
        context_tag_dict.update(tags_add)
        context_answer_dict.update(answer_add)
        context_question_dict.update(question_add)

        if evaluate:
            context_caption_dict = dict(list(context_caption_dict.items())[-NUM_TEST_EXAMPLES:])
            context_tag_dict = dict(list(context_tag_dict.items())[-NUM_TEST_EXAMPLES:])
            context_answer_dict = dict(list(context_answer_dict.items())[-NUM_TEST_EXAMPLES:])
            context_question_dict = dict(list(context_question_dict.items())[-NUM_TEST_EXAMPLES:])

        return context_caption_dict, context_tag_dict, context_answer_dict, context_question_dict


class Pipeline:
    """
    Main inference class
    """

    def __init__(self):
        # Tagging model setup
        segment_model = DetrForSegmentation.from_pretrained(tag_model, use_pretrained_backbone=False)
        self.segment = pipeline(
            "image-segmentation", model=segment_model, feature_extractor=DetrFeatureExtractor.from_pretrained(tag_model)
        )
        self.tags = []

        # Caption model setup
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model)
        self.caption_feature_extractor = ViTFeatureExtractor.from_pretrained(caption_model)
        self.caption_tokenizer = AutoTokenizer.from_pretrained(caption_model)
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CLIP Setup
        self.clip_session = InferenceSession(str(clip_onnx))
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor)

    def predict_caption(self, image):
        pixel_values = self.caption_feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        output_ids = self.caption_model.generate(pixel_values, **gen_kwargs)

        preds = self.caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]

    def predict(self, image: Union[str, Path, Image.Image], question: Union[str, Path]) -> str:
        if not isinstance(image, Image.Image):
            image_pil = Image.open(image)
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert(mode="RGB")
        else:
            image_pil = image
        if isinstance(question, Path) | os.path.exists(question):
            with open(question, "r") as f:
                question_str = f.readline()
        else:
            question_str = question

        # Generating image tag(s)
        for dic in self.segment(image_pil):
            self.tags.append(dic["label"])
        if not self.tags:
            self.tags.append("")
        tag_info: Dict[int, List[str]] = {img_id: ", ".join(self.tags)}

        # Generating image caption
        caption = self.predict_caption(image_pil)
        if not caption:
            caption = ""
        caption_info: Dict[int, str] = {img_id: caption}

        # Generating image/question features
        inputs = self.clip_processor(text=[question_str], images=image_pil, return_tensors="np", padding=True)
        # for i in session.get_outputs(): print(i.name)
        outputs = self.clip_session.run(
            output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
        )

        # Generating context idxs
        context_idxs: Dict[str, str] = {"0": str(img_id) + "<->" + str(question_id)}

        # Answering question
        questions = {"questions": [{"image_id": img_id, "question": question_str, "question_id": question_id}]}
        okvqa = PICa_OKVQA(
            caption_info, tag_info, questions, context_idxs, outputs[2], outputs[3]
        )  # Have to initialize here because necessary objects need to be generated
        answer = okvqa.answer_gen()
        # rationale = okvqa.rationale(answer)

        return answer  # + " because " + rationale

    def evaluate(self):
        okvqa = PICa_OKVQA(
            evaluate=True,
        )
        acc = okvqa.answer_gen()
        print(acc)
        return acc


# Running model
def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)

    args = parser.parse_args()

    # Answering question
    pipeline = Pipeline()
    pred_str = pipeline.predict(args.image, args.question)

    print(pred_str)


if __name__ == "__main__":
    main()
