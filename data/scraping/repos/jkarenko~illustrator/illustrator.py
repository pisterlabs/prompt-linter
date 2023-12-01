#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import base64
from random import choice
from openai import OpenAI


def encode_image(image_path):
  print(f"Encoding image: {image_path}")
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

TEST = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:\n"
STYLE_LIST = ['vivid', 'dark', 'bright', 'surreal', 'realistic', 'cartoonish', 'painting-like', 'photorealistic', 'impressionistic', 'abstract', 'minimalist', 'simplistic', 'complex', 'busy', 'chaotic', 'calm', 'peaceful', 'tranquil', 'serene', 'relaxing', 'tense', 'stressful', 'scary', 'frightening', 'terrifying', 'horrifying', 'creepy', 'eerie', 'unsettling', 'disturbing', 'upsetting', 'sad', 'depressing', 'melancholic', 'melancholy', 'happy', 'joyful', 'cheerful', 'playful', 'fun', 'funny', 'humorous', 'hilarious', 'amusing', 'entertaining', 'exciting', 'thrilling', 'adventurous', 'romantic', 'sexy', 'erotic', 'sensual', 'sexual', 'sophisticated', 'elegant', 'classy', 'stylish', 'fashionable', 'trendy', 'modern', 'futuristic', 'retro', 'nostalgic', 'old-fashioned', 'traditional', 'historic', 'ancient', 'mythological', 'fantasy', 'magical', 'mystical', 'mysterious', 'science-fiction', 'sci-fi', 'futuristic', 'utopian', 'dystopian', 'apocalyptic', 'post-apocalyptic', 'cyberpunk', 'steampunk', 'gothic', 'medieval', 'western', 'noir', 'film-noir', 'noirish', 'gritty', 'dark', 'grim', 'gruesome', 'macabre', 'morbid', 'grotesque', 'horror', 'horror-like', 'horror-themed', 'horror-inspired']

client = OpenAI()

# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument("--scene", default="", help="Scene description")
parser.add_argument("--style", nargs="?", default="fitting the scene", help="Style description, e.g. " + ", ".join(STYLE_LIST))
parser.add_argument("--dalle", choices=['2', '3'], default='3', help="Choose Dall-e version, either '2' or '3'")
parser.add_argument("--optimized", action="store_true", default=False, help="Create an additional optimized prompt for Dall-e")
parser.add_argument("--environment", type=str, help="Environment override, e.g. 'a fantasy world'")
parser.add_argument("--reference-image", type=str, default="", help="Reference image for the illustration")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")

visual_group = parser.add_argument_group('Image to text settings')
visual_group.add_argument("--image-url", help="Use image at URL as scene descriptor")
visual_group.add_argument("--image-file", help="Use image file as scene descriptor")
visual_group.add_argument("--image-quality", choices=['high', 'low'], default='auto', help="Image quality used during image to text, either 'high' or 'low'")

# Create dalle-3 group
dalle_3_group = parser.add_argument_group('Dall-e 3 specific settings')
dalle_3_group.add_argument("--hd", action="store_true", default=False, help="High definition mode for more detailed images")
dalle_3_group.add_argument("--detail", choices=['vivid', 'natural'], default='vivid', help="Detail level, either 'vivid' or 'natural'")

parser.add_argument("--size", choices=["1", "2", "3"], default='3', help="1=portrait (1024x1792) or small (256x256), 2=landscape (1792x1024) or medium (512x512), 3=square (1024x1024), default=3")
parser.add_argument("--no-test", action="store_true", help="Disable test mode prompt hack")
args = parser.parse_args()

# Check for mutual exclusivity
dalle_3_args = [args.hd, args.detail, args.dalle == '3']
dalle_2_args = [args.dalle == '2']

if any(dalle_3_args) and any(dalle_2_args):
    parser.error("Dall-e 2 and Dall-e 3 image settings are mutually exclusive")

if args.style.lower() == 'random':
  args.style = choice(STYLE_LIST)

sizes_dalle_3 = {
  '1': '1792x1024',
  '2': '1024x1792',
  '3': '1024x1024',
}

sizes_dalle_2 = {
  '1': '256x256',
  '2': '512x512',
  '3': '1024x1024',
}

sizes = sizes_dalle_3 if args.dalle == '3' else sizes_dalle_2
encoded_image = encode_image(args.image_file) if args.image_file else None
image_url = f"data:image/jpeg;base64,{encoded_image}" if encoded_image else args.image_url
encoded_reference_image = None
if not args.reference_image.startswith('http'):
  encoded_reference_image = encode_image(args.reference_image) if args.reference_image else None
reference_image_url = f"data:image/jpeg;base64,{encoded_reference_image}" if encoded_reference_image else args.reference_image
# print(image_url)


def describe_image(image_url=None):
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": f"Describe this image and all its entities in a non-provocative, non-sexualized way. Describe everything in great detail (person's/creature's gender presentation, ethnicity, age, height(short/average/tall), size(small/average/large)). Pay attention to character positions and facings. Start immediately with the description."},
          {
            "type": "image_url",
            "image_url": {
              "url": f"{image_url}",
              "detail": f"{args.image_quality}",
            },
          },
        ],
      }
    ],
    max_tokens=1000,
  )
  return response.choices[0].message.content


def describe_style(image_url=None):
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": f"Describe concisely the visual style only of this image in terms of setting (sci-fi, western, fantasy etc), colors (warm, cold, monochrome, faded, orange and teal, grayish etc), backdrop (outdoors, indoors, space, nature, grocery store etc)."},
          {
            "type": "image_url",
            "image_url": {
              "url": f"{image_url}",
              "detail": f"{args.image_quality}",
            },
          },
        ],
      }
    ],
    max_tokens=1000,
  )
  return response.choices[0].message.content

image_description = None
reference_description = None

if args.image_url or args.image_file:
  print(f"Describing image...")
  image_description = describe_image(image_url=image_url)

if args.reference_image:
  print(f"Describing reference image...")
  reference_description = describe_style(image_url=reference_image_url)

environment_description = None

if args.environment:
  print(f"Describing override environment...")
  response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {
        "role": "user",
        "content": f"Describe the visual style of {args.environment} including but not limited to art, camera angle, view point etc",}
    ],
    max_tokens=1000,
  )
  environment_description = response.choices[0].message.content
if not environment_description and reference_description:
  environment_description = reference_description

if environment_description:
  print(f"Re-describing scene with override environment...")
  response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {
        "role": "user",
        "content": f"Environment:\n{environment_description}\n\nUsing the above description revise the following scene:\n{image_description}. Start immediately with the description, do not add any additional text, do not ask questions, do not explain what changed and only change the backdrop and leave all action and framing of characters intact.",
      }
    ],
    max_tokens=1000,
  )
  image_description = response.choices[0].message.content


scene_description = f"\nOverview:\n{args.scene}\nDetails:\n{image_description}"

optimized_prompt = None
if args.optimized:
  print(f"Creating optimized prompt for Dall-e...")
  completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {"role": "system", "content": f"You are a system that creates optimized dall-e 3 prompts."},
      {"role": "user", "content": f"Describe this creation so Dall-e 3 can recreate it accurately as a {args.style} style image: {scene_description}. More emphasis on overview if provided. Only use visual facts, opinions and describe it as if it was created in the style of {args.style}."},
    ]
  )
  optimized_prompt = completion.choices[0].message.content

scene_prompt = ""
if args.optimized:
  scene_prompt = optimized_prompt
else:
  scene_prompt = f"\nScene:\n{scene_description}\n\nStyle:\n{args.style}\n\n{scene_prompt}"
scene_prompt = TEST + scene_prompt if not args.no_test else scene_prompt
if args.debug: print(scene_prompt)

response = client.moderations.create(input=scene_prompt)
moderation = response.results[0]
if moderation.flagged:
  print(f"\n{[category for category, value in moderation.categories.items() if value]}")
  exit()

dalle_2_params = {
  "model": "dall-e-2",
  "prompt": scene_prompt,
  "size": sizes[args.size],
  "n": 1,
}

dalle_3_params = {
  "model": "dall-e-3",
  "prompt": scene_prompt,
  "style": args.detail,
  "size": sizes[args.size],
  "quality": "hd" if args.hd else "standard",
  "n": 1,
}

image_params = dalle_3_params if args.dalle == '3' else dalle_2_params
# print(f"\n{image_params}")

print(f"Generating image...")
response = client.images.generate(
  **dalle_3_params if args.dalle == '3' else dalle_2_params
)

image_url = response.data[0].url
# print(f"\nOriginal image:\n{args.image_url}")
print(f"\nGenerated image:\n{image_url}")
