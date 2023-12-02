"""The Edge of Reality
"""
from random import randint
from uuid import uuid4
from typing import List, Dict, Optional

import openai
from PIL import Image

from config import FRAME_SIZE, FRAME_COUNT, VARIATIONS
from world_generator.utils.dalle import (
    download_image,
    generate_frame_variation,
    generate_last_frame_url,
    generate_next_frame_url
)
from world_generator.utils.decorators import print_execution_time
from world_generator.utils.gpt import get_prompt_variations
from world_generator.utils.types import WorldFrame


def merge_frames(frames: List[WorldFrame], dst: str) -> WorldFrame:
    """Merging all frames together, the last frame is used differently #TODO: Explain better
    """
    last_frame = frames.pop()

    # Create a new image that will contain all of the merged images
    init_width = sum([i['image'].size[0] for i in frames])
    total_width = (init_width / 2) + FRAME_SIZE
    new_image = Image.new('RGB', (int(total_width), FRAME_SIZE))

    # Paste each image into the new image
    x_offset = 0
    for i, frame in enumerate(frames):
        new_image.paste(frame['image'], (x_offset, 0))
        x_offset += frame['image'].size[0] - (FRAME_SIZE // 2)

    x_offset = x_offset + (FRAME_SIZE // 4)
    new_image.paste(last_frame['image'], (x_offset, 0))

    # Save the merged image
    new_image.save(dst)
    print('Frames merged for first iteration')
    return WorldFrame(path=dst, image=new_image)


@print_execution_time
def generate_new_world(settings: Dict, dst: str) -> None:
    """
    """
    frames: List[WorldFrame] = []
    last_path = ''

    for i in range(FRAME_COUNT):
        print(f'\nGenereting frame #{i}...')
        if i == 0:
            response = openai.Image.create(**settings)
            url = response['data'][0]['url']
        elif i < FRAME_COUNT - 1:
            url = generate_next_frame_url(last_path, **settings)
        else:
            url = generate_last_frame_url(frames[0]['path'], frames[-1]['path'], **settings)

        frame = download_image(url, dst=f'.tmpfiles/{i}.png')
        last_path = frame['path']
        frames.append(frame)


    merged_frame = merge_frames(frames, dst=f'{dst}/0.png')


    # Variations:
    # prompt_variations = get_prompt_variations(settings['prompt'], count=VARIATIONS)
    width, _ = merged_frame['image'].size

    latest = merged_frame['image']
    total = 1
    for j in range(10):
        for i in range(VARIATIONS):
            print(f'Generating variation {i + 1}')
            # s = settings.copy()
            # s['prompt'] = prompt
            new_variation = latest

            x1 = i*(width // VARIATIONS)
            y1 = 0
            x2 = x1 + 1024
            y2 = 1024

            # Crop the image to get the current frame
            frame = new_variation.crop((x1, y1, x2, y2))
            # frame.save(f'frame_{i}_before.png')
            v_url = generate_frame_variation(frame=frame, **settings)
            v_frame = download_image(v_url, dst=f'.tmpfiles/frame_{i}_after.png')
            new_variation.paste(v_frame['image'], (x1, 0))
            new_variation.save(f'{dst}/{total}.png')
            latest = new_variation
            total += 1