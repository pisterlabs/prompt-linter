import os
import openai
from transformers import pipeline, CLIPModel, CLIPProcessor, CLIPTokenizer
from datasets import load_dataset
from tqdm import tqdm
from data.winoground import WinogroundDataset

def make_prompt(bounding_boxes, threshold=0.9, img_width=1920, img_height=1280):
    full_prompt = "Describe this image:\nObjects in this image:\n"
    large_width = img_width // 3
    large_height = img_height // 3
    med_width = img_width // 4
    med_height = img_height // 4
    num_objects = 0

    for box in bounding_boxes:
        if box["score"] < threshold:
            continue
        num_objects += 1
        box_width = box["box"]["xmax"] - box["box"]["xmin"]
        x_center = (box["box"]["xmax"] + box["box"]["xmin"]) / 2
        box_height = box["box"]["ymax"] - box["box"]["ymin"]
        y_center = (box["box"]["ymax"] + box["box"]["ymin"]) / 2

        prompt = f"{box['label']} is at "

        if y_center < large_height:
            prompt += "bottom "
        elif y_center < 2 * large_height:
            prompt += "middle "
        else:
            prompt += "top "

        if x_center < large_width:
            prompt += "left "
        elif x_center < 2 * large_width:
            if "middle" not in prompt:
                prompt += "middle "
        else:
            prompt += "right "

        if box_width > large_width or box_height > large_height:
            prompt += "and is large in the image.\n"
        elif box_width > med_width or box_height > med_height:
            prompt += "and is moderate in the image.\n"
        else:
            prompt += "and is small in the image.\n"
        
        full_prompt += f"{prompt}\n"

    full_prompt += "Describe this image:\n"
    return full_prompt, num_objects


def generate_gpt3(prompt):
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0,
        n=1,
    )
    return completion.choices[0].text

def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

def group_correct(result):
    return image_correct(result) and text_correct(result)

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    winoground_dataset = WinogroundDataset()
    winoground_clip_scores = []
    visual_clues_scores = []
    winoground = load_dataset("facebook/winoground", use_auth_token="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")["test"]
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    detector = pipeline("object-detection")

    for i, example in enumerate(tqdm(winoground)):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
        input_c0_i0 = clip_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
        input_c1_i0 = clip_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
        input_c0_i1 = clip_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")
        input_c1_i1 = clip_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")

        bounding_boxes_i0 = detector(example["image_0"])
        bounding_boxes_i1 = detector(example["image_1"])
        prompt_i0, num_objects_i0 = make_prompt(bounding_boxes_i0)
        prompt_i1, num_objects_i1 = make_prompt(bounding_boxes_i1)

        if num_objects_i0 > 0:
            gpt_desc_i0 = generate_gpt3(prompt_i0)
            # find similarity between the text embedding and original embeddings
            gpt_c0_i0 = clip_processor(text=[gpt_desc_i0], images=[example["image_0"].convert("RGB")], return_tensors="pt")
            gpt_c0_i1 = clip_processor(text=[gpt_desc_i0], images=[example["image_1"].convert("RGB")], return_tensors="pt")
            #orig_c0 = clip_processor(text=[example["caption_0"]], images=None, return_tensors="pt")
            #orig_c1 = clip_processor(text=[example["caption_1"]], images=None, return_tensors="pt")

            output_c0_gpt = model.get_text_features(**input_c0)
            output_c1_gpt = model.get_text_features(**input_c1)
            orig_output_c0 = model.get_text_features(**orig_input_c0)
            orig_output_c1 = model.get_text_features(**orig_input_c1)

            gpt_output_c0_i0 = clip_model(**gpt_c0_i0)
            gpt_output_c0_i1 = clip_model(**gpt_c0_i1)
            
            gpt_clip_score_c0_i0 = gpt_output_c0_i0.logits_per_image.item()
            gpt_clip_score_c0_i1 = gpt_output_c0_i1.logits_per_image.item()
        else:
            gpt_clip_score_c0_i0 = 0
            gpt_clip_score_c0_i1 = 0

        if num_objects_i1 > 0:
            gpt_desc_i1 = generate_gpt3(prompt_i1)
            gpt_c1_i0 = clip_processor(text=[gpt_desc_i1], images=[example["image_0"].convert("RGB")], return_tensors="pt")
            gpt_c1_i1 = clip_processor(text=[gpt_desc_i1], images=[example["image_1"].convert("RGB")], return_tensors="pt")
            
            gpt_output_c1_i0 = clip_model(**gpt_c1_i0)
            gpt_output_c1_i1 = clip_model(**gpt_c1_i1)

            gpt_clip_score_c1_i0 = gpt_output_c1_i0.logits_per_image.item()
            gpt_clip_score_c1_i1 = gpt_output_c1_i1.logits_per_image.item()
        else:
            gpt_clip_score_c1_i0 = 0
            gpt_clip_score_c1_i1 = 0

        output_c0_i0 = clip_model(**input_c0_i0)
        output_c0_i1 = clip_model(**input_c0_i1)
        clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
        clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
        output_c1_i0 = clip_model(**input_c1_i0)
        output_c1_i1 = clip_model(**input_c1_i1)
        
        clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
        clip_score_c1_i1 = output_c1_i1.logits_per_image.item()


        final_score_c0_i0 = clip_score_c0_i0 + gpt_clip_score_c0_i0
        final_score_c1_i0 = clip_score_c1_i0 + gpt_clip_score_c1_i0
        final_score_c0_i1 = clip_score_c0_i1 + gpt_clip_score_c0_i1
        final_score_c1_i1 = clip_score_c1_i1 + gpt_clip_score_c1_i1

        winoground_clip_scores.append({"id": example["id"], "c0_i0": clip_score_c0_i0, "c1_i0": clip_score_c1_i0, "c0_i1": clip_score_c0_i1, "c1_i1": clip_score_c1_i1})
        visual_clues_scores.append({"id" : example["id"], "c0_i0": final_score_c0_i0, "c0_i1": final_score_c0_i1, "c1_i0": final_score_c1_i0, "c1_i1": final_score_c1_i1})

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    text_correct_clues = 0
    image_correct_clues = 0
    group_correct_clues = 0
    for result in visual_clues_scores:
        text_correct_clues += 1 if text_correct(result) else 0
        image_correct_clues += 1 if image_correct(result) else 0
        group_correct_clues += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)
    print("text score:", text_correct_count/denominator)
    print("image score:", image_correct_count/denominator)
    print("group score:", group_correct_count/denominator)

    print("text score with clues:", text_correct_clues/denominator)
    print("image score with clues:", image_correct_clues/denominator)
    print("group score with clues:", group_correct_clues/denominator)


