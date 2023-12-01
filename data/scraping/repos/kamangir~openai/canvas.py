from io import BytesIO
from PIL import Image
from enum import Enum
from tqdm import tqdm
import requests
import numpy as np
from openai_cli.DALLE.brush import RandomWalkBrush, TilingBrush
from openai_cli import NAME, VERSION
from abcli.modules.objects import signature as object_signature
from abcli.modules.host import signature as host_signature
from abcli.modules.host import is_jupyter
from abcli.plugins.graphics import add_signature
from abcli import file
import abcli.logging
from abcli.logging import crash_report
import logging

logger = logging.getLogger()


class Canvas(object):
    def __init__(
        self,
        shape=(2048, 4096),
        verbose=False,
        debug_mode=False,
        dryrun=False,
        content=None,
        source="",
        brush_kind="tiling",
        brush_size=256,
    ):
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.dryrun = dryrun
        self.is_jupyter = is_jupyter()
        self.source = source
        self.content = content

        if content is not None:
            shape = Canvas.shape(
                content,
                brush_kind,
                brush_size,
            )

        self.height, self.width = shape

        self.allocated = not dryrun

        self.image = Image.new(
            "RGB",
            (
                0 if dryrun else self.width,
                0 if dryrun else self.height,
            ),
            (0, 0, 0),
        )
        self.mask = Image.new(
            "L",
            (self.width, self.height),
            (0,),
        )

        logger.info(
            "DALL-E Canvas({}x{}): {}allocated".format(
                self.height,
                self.width,
                "" if self.allocated else "not ",
            )
        )

    def add_signature(self, image):
        return Image.fromarray(
            add_signature(
                np.array(image),
                [" | ".join(object_signature())],
                [
                    "{} : {}".format(
                        self.content[0] if len(self.content) >= 1 else "",
                        self.source,
                    )
                ]
                + [
                    " | ".join([f"{NAME}-{VERSION}"] + host_signature()),
                ],
            )
        )

    def box(self):
        indices = np.nonzero(np.array(self.mask) == 255)

        left = indices[1].min()
        top = indices[0].min()
        right = indices[1].max()
        bottom = indices[0].max()

        return (left, top, right, bottom)

    def create_brush(
        self,
        brush_kind="tiling",
        brush_size=256,
    ):
        if brush_kind == "tiling":
            return TilingBrush(self, brush_size, brush_size)
        elif brush_kind == "randomwalk":
            return RandomWalkBrush(self, brush_size, brush_size)
        else:
            raise ValueError(
                f"-DALL-E: Canvas: create_brush: {brush_kind}: kind not found."
            )

    def paint(self, brush, prompt):
        import openai

        box = (
            brush.cursor[0] - brush.width // 2,  # left,
            brush.cursor[1] - brush.height // 2,  # top,
            brush.cursor[0] + brush.width // 2,  # right,
            brush.cursor[1] + brush.height // 2,  # bottom,
        )

        image_ = self.image.crop(box)
        image_byte_stream = BytesIO()
        image_.save(image_byte_stream, format="PNG")
        image_byte_array = image_byte_stream.getvalue()

        mask_ = image_.copy().convert("RGBA")
        mask_.putalpha(self.mask.crop(box))
        mask_byte_stream = BytesIO()
        mask_.save(mask_byte_stream, format="PNG")
        mask_byte_array = mask_byte_stream.getvalue()

        if not self.dryrun:
            try:
                response = openai.Image.create_edit(
                    image=image_byte_array,
                    mask=mask_byte_array,
                    prompt=prompt,
                    n=1,
                    size=f"{brush.width}x{brush.height}",
                )

                image_url = response["data"][0]["url"]
                if self.verbose:
                    logger.info(f"Canvas.paint: received {image_url}")

                response = requests.get(image_url)
                image_data = response.content
                image__ = Image.open(BytesIO(image_data))
                if self.verbose:
                    logger.info(
                        f"Canvas.paint: downloaded {image__.size}, {image__.mode}"
                    )

                self.image.paste(image__, box)
            except:
                crash_report(f"{NAME}: paint({prompt})")

        self.mask.paste(
            Image.new(
                "L",
                (brush.width, brush.height),
                (255,),
            ),
            box,
        )

        if self.is_jupyter and self.verbose:
            from IPython.display import display, clear_output

            clear_output(wait=True)

            if self.debug_mode:
                image = Image.new("RGB", (3 * brush.width, brush.height))
                image.paste(image_, (0, 0))
                image.paste(mask_, (brush.width, 0))
                image.paste(image__, (2 * brush.width, 0))
                display(self.add_signature(image))

                image = Image.new("RGB", (2 * self.width, self.height))
                image.paste(self.mask, (0, 0))
                image.paste(self.image, (self.width, 0))
                display(self.add_signature(image))
            else:
                display(self.add_signature(self.image.crop(self.box())))

        return self

    def render_text(
        self,
        brush,
        content,
        filename,
    ):
        content = [line for line in content if line]

        logger.info(f"Canvas.render_text: {len(content)} line(s).")

        for index in tqdm(range(len(content))):
            if self.verbose:
                logger.info(f"DALL-E: {content[index]}")

            self.paint(brush, content[index])

            brush.move(self)

            if self.verbose:
                self.save(filename)

        self.save(filename)

        return self

    def save(self, filename):
        box = self.box()

        mask = self.mask.crop(box)
        image = self.image.crop(box)

        image = self.add_signature(image)

        image.save(filename)
        mask.save(file.add_postfix(filename, "mask"))

        if self.debug_mode and self.is_jupyter:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(image)

        logger.info(f"Canvas -> {filename}")

        file.save_json(
            file.set_extension(filename, "json"),
            {
                "content": self.content,
                "source": self.source,
                "height": self.height,
                "width": self.width,
                "generator": f"DALL-E-{VERSION}",
            },
        )

    @staticmethod
    def shape(
        content,
        brush_kind="tiling",
        brush_size=256,
        margin=0.0,
    ):
        canvas = Canvas(
            (25000, 25000),
            dryrun=True,
        )

        brush = canvas.create_brush(
            brush_kind,
            brush_size,
        )

        content = [line for line in content if line]

        for index in tqdm(range(len(content))):
            canvas.paint(brush, content[index])
            brush.move(canvas)

        left, top, right, bottom = canvas.box()

        plus_margin = 1 + 2 * margin

        height = int(
            2
            * max(
                canvas.height // 2 - top,
                bottom - canvas.height // 2,
            )
            * plus_margin
        )
        width = int(
            2
            * max(
                canvas.width // 2 - left,
                right - canvas.height // 2,
            )
            * plus_margin
        )

        logger.info(
            f"Canvas.shape: {len(content)} line(s) @ {brush_kind} x {brush_size}: {height}x{width}"
        )

        return height, width
