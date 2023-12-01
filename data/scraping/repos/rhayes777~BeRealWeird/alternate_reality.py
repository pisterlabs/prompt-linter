import shutil
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
import openai

from bereal_gpt.described_memory import DescribedMemory
from bereal_gpt.weird_image import WeirdImage


def _generate_image(prompt: str, image_path: Path, size="1024x1024", ratio=(1.5 / 2)):
    if not image_path.exists():
        image = openai.Image.create(
            prompt=prompt,
            size=size,
        )
        url = image.data[0]["url"]

        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(image_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)

    image = Image.open(image_path)
    width, height = image.size
    new_width = int(height * ratio)
    left = int((width - new_width) / 2)
    return image.crop((left, 0, left + new_width, height))


class AlternateReality(WeirdImage):
    def __init__(self, described_memory: DescribedMemory, style=None):
        self.described_memory = described_memory
        self.style = style

    @property
    def _directory(self):
        directory = Path("alternate_reality") / self.style.replace(" ", "_") / str(self.memory_day())
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def image_path(self):
        return self._directory.with_suffix(".png")

    def primary_image(self):
        return _generate_image(
            f"A photo containing {self.described_memory.primary_description()}. {self.style}",
            self._directory / "alternate_primary.png",
        )

    def secondary_image(self):
        return _generate_image(
            f"A photo containing {self.described_memory.secondary_description()}. The photo is taken as a selfie. {self.style}",
            self._directory / "alternate_secondary.png",
            size="256x256",
        )

    @classmethod
    def _from_directory(cls, directory, style=None):
        return [cls(memory, style=style) for memory in DescribedMemory.from_directory(directory)]

    def memory_day(self):
        return self.described_memory.memory.memory_day()

    def comparison_image(self):
        alternate_reality_image = self.image()
        described = self.described_memory
        memory = described.memory
        real_image = memory.image()
        # Resize images to the same size if needed
        real_image = real_image.resize(alternate_reality_image.size)

        # Define the panel size (considering image sizes and text space)
        panel_size = (real_image.width * 2, real_image.height + 60)  # 60 pixels for text

        # Create a new image with white background
        panel = Image.new('RGB', panel_size, (255, 255, 255))

        # Paste the images into the panel
        panel.paste(real_image, (0, 60))  # First image at (0,60)
        panel.paste(alternate_reality_image, (real_image.width, 60))  # Second image beside the first one

        # Add text
        draw = ImageDraw.Draw(panel)
        font = ImageFont.truetype("Arial Unicode.ttf", 15)
        draw.text(
            (10, 10),
            f"{described.primary_description()} ({described.secondary_description()})",
            fill="black",
            font=font
        )

        return panel
