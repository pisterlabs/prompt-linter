# MIT License

# Copyright (c) 2023 David Rice

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import base64
import logging

import openai
import pygame
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk import client

from log_config import get_logger_name

logger = logging.getLogger(get_logger_name())


class ArtistCreation:
    """
    Class representing a full "creation" by the A.R.T.I.S.T. system, i.e., the image
    and its corresponding verse.
    """

    def __init__(
        self,
        img: pygame.Surface,
        verse_lines: list[str],
        prompt: str,
        is_daydream: bool,
    ) -> None:
        self.img = img
        self.verse_lines = verse_lines
        self.prompt = prompt
        self.is_daydream = is_daydream


class ArtistCanvas:
    """
    Class representing the visible surface on which the ArtistCreation object
    will be rendered.
    """

    def __init__(
        self,
        width: int,
        height: int,
        horiz_margin: int,
        vert_margin: int,
        verse_font_name: str,
        verse_font_max_size: int,
        verse_line_spacing: int,
    ) -> None:
        self._width = width
        self._height = height

        self._horiz_margin = horiz_margin
        self._vert_margin = vert_margin

        self._verse_font_name = verse_font_name
        self._verse_font_max_size = verse_font_max_size
        self._verse_line_spacing = verse_line_spacing

        self._surface = pygame.Surface(size=(width, height))

    def _get_verse_font_size(self, verse_lines: list[str], max_verse_width: int) -> int:
        font_obj = pygame.font.SysFont(self._verse_font_name, self._verse_font_max_size)
        longest_line_size = 0

        # Need to check pizel size of each line to account for
        # proprtional fonts. Assumes that size scales linearly.
        for line in verse_lines:
            text_size = font_obj.size(line)
            if text_size[0] > longest_line_size:
                longest_line_size = text_size[0]
                longest_line = line

        font_size = self._verse_font_max_size
        will_fit = False

        while not will_fit:
            font_obj = pygame.font.SysFont(self._verse_font_name, font_size)

            text_size = font_obj.size(longest_line)

            if text_size[0] < max_verse_width:
                will_fit = True
            else:
                font_size -= 2

        return font_size

    def _get_verse_total_height(
        self, verse_lines: list[str], verse_font_size: int
    ) -> int:
        font_obj = pygame.font.SysFont(self._verse_font_name, verse_font_size)

        total_height = 0

        for line in verse_lines:
            text_size = font_obj.size(line)

            total_height += text_size[1]
            total_height += self._verse_line_spacing

        total_height -= self._verse_line_spacing

        return total_height

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def clear(self) -> None:
        self._surface.fill(color=pygame.Color("black"))

    def render_creation(self, creation: ArtistCreation, img_side: str) -> None:
        self.clear()

        img_width = creation.img.get_width()

        if img_side.lower() == "left":
            img_x = self._horiz_margin
            verse_x = self._horiz_margin + img_width + self._horiz_margin
        elif img_side.lower() == "right":
            img_x = self._width - self._horiz_margin - img_width
            verse_x = self._horiz_margin
        else:
            raise ValueError("img_side must be either 'left' or 'right'")

        # Draw the image
        self._surface.blit(source=creation.img, dest=(img_x, self._vert_margin))

        max_verse_width = (self._width - img_width) - (self._horiz_margin * 3)
        verse_font_size = self._get_verse_font_size(
            creation.verse_lines, max_verse_width
        )

        total_height = self._get_verse_total_height(
            creation.verse_lines, verse_font_size
        )
        offset = -total_height // 2

        font_obj = pygame.font.SysFont(self._verse_font_name, verse_font_size)

        for line in creation.verse_lines:
            text_surface = font_obj.render(line, True, pygame.Color("white"))
            self._surface.blit(
                source=text_surface, dest=(verse_x, (self._height // 2) + offset)
            )

            offset += int(total_height / len(creation.verse_lines))


class StatusScreen:
    """
    Class representing the status screen displayed when A.R.T.I.S.T. is
    waiting for input or generating a new creation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        font_name: str,
        heading1_size: int,
        heading2_size: int,
        status_size: int,
        vert_margin: int,
    ) -> None:
        self._width = width
        self._height = height
        self._font_name = font_name
        self._heading1_size = heading1_size
        self._heading2_size = heading2_size
        self._status_size = status_size
        self._vert_margin = vert_margin

        self._surface = pygame.Surface(size=(width, height))

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def render_status(self, text: str) -> None:
        self._surface.fill(pygame.Color("black"))

        font = pygame.font.SysFont(self._font_name, self._heading1_size)
        heading1 = "A.R.T.I.S.T."
        x_pos = int(self._surface.get_width() / 2 - font.size(heading1)[0] / 2)
        y_pos = self._vert_margin
        text_surface = font.render(heading1, True, pygame.Color("white"))
        self._surface.blit(text_surface, (x_pos, y_pos))

        heading1_height = font.size(heading1)[1]

        font = pygame.font.SysFont(self._font_name, self._heading2_size)
        heading2 = "Audio-Responsive Transformative Imagination Synthesis Technology"
        x_pos = int(self._surface.get_width() / 2 - font.size(heading2)[0] / 2)
        y_pos += heading1_height
        text_surface = font.render(heading2, True, pygame.Color("white"))
        self._surface.blit(text_surface, (x_pos, y_pos))

        font = pygame.font.SysFont(self._font_name, self._status_size)
        x_pos = int(self._surface.get_width() / 2 - font.size(text)[0] / 2)
        y_pos = int(self._surface.get_height() / 2 - font.size(text)[1] / 2)
        text_surface = font.render(text, True, pygame.Color("white"))
        self._surface.blit(text_surface, (x_pos, y_pos))


class SDXLCreator:
    def __init__(
        self,
        api_key: str,
        img_width: int,
        img_height: int,
        steps: int,
        cfg_scale: float,
    ) -> None:
        self.api_key = api_key
        self.img_width = img_width
        self.img_height = img_height
        self.steps = steps
        self.cfg_scale = cfg_scale

        self._stability_client = client.StabilityInference(
            key=self.api_key,
            engine="stable-diffusion-xl-1024-v1-0",
        )

    def generate_image_data(self, prompt: str) -> bytes:
        response = self._stability_client.generate(
            prompt=prompt,
            width=self.img_width,
            height=self.img_height,
            steps=self.steps,
            cfg_scale=self.cfg_scale,
        )

        for r in response:
            for artifact in r.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    logger.error("Content filter triggered")
                    raise RuntimeError("Content filter triggered")
                else:
                    return artifact.binary

        raise RuntimeError("No artifact returned")


class DallE2Creator:
    def __init__(self, api_key: str, img_width: int, img_height: int) -> None:
        self.api_key = api_key
        self.img_width = img_width
        self.img_height = img_height

    def generate_image_data(self, prompt: str) -> bytes:
        img_size = f"{self.img_width}x{self.img_height}"

        try:
            response = openai.Image.create(
                api_key=self.api_key,
                prompt=prompt,
                size=img_size,
                response_format="b64_json",
                user="A.R.T.I.S.T.",
            )
        except Exception as e:
            logger.error(f"Image creation response: {response}")
            logger.exception(e)
            raise

        return base64.b64decode(response["data"][0]["b64_json"])
