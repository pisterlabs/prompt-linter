import io
import os

import numpy as np
from signalbot import Context
from PIL import Image
from openai import InvalidRequestError

from safe_command import SafeCommand
from dalle import create_variations
from utils import resize_image, save_b64_images
from translate import from_english


class VariationsCommand(SafeCommand):
    
    VARIATIONS_COUNT = int(os.environ.get("VARIATIONS_COUNT", 1))
    
    def describe(self) -> str:
        return "Respond with image variations created with DALL-E"

    async def handle_save(self, c: Context):
        prompt = c.message.text
        attachments = c.message.base64_attachments
        
        # must be image without text to trigger command
        if not (len(attachments) > 0 and not prompt):
            return
        
        # single file must be attached
        if not (len(attachments) == 1 and attachments[0].content_type.split('/')[0] == 'image'):
            await c.react('🤔')
            return
        
        await c.react('👍')
        await c.start_typing()
        
        attachment = attachments[0]
        await c.fetch_attachment_data(attachment)
        
        image = np.array(resize_image(Image.open(io.BytesIO(attachment.data)), length=1024)) 
        
        try:
            variations = create_variations(image, self.VARIATIONS_COUNT)
        except InvalidRequestError:
            await c.send(from_english('Generating this content was blocked by OpenAI 😕'))
            return
        finally:
            await c.stop_typing()
        
        save_b64_images(variations, 'variation')
        
        await c.send(
            from_english('These are variants of the uploaded photo'),
            base64_attachments=variations
        )
