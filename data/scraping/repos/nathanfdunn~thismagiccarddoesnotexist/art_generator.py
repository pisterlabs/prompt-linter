import openai
from PIL import Image, ImageOps
import requests
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
import cloudinary.uploader
import cloudinary

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

def create_mask(input_image: str):
    # Open the image file
    img = Image.open(input_image)
    # Get the dimensions of the image
    width, height = img.size
    # Scale the image down to 2/3 of its size
    new_size = (int(width * 2/3), int(height * 2/3))
    img = img.resize(new_size)
    # Create a new image with the original dimensions and transparent background
    new_img = Image.new('RGBA', (width, height))
    # Paste the scaled image into the center of the new image
    new_img.paste(img, ((width - new_size[0])//2,
                      (height - new_size[1])//2))
    # left_padding = (width - new_size[0])//2
    # right_padding = width - new_size[0] - left_padding
    mask_location = input_image.split('.')[0] + '_padded.png'
    new_img.save(input_image.split('.')[0] + '_padded.png')
    return mask_location

def outpaint(mask: str, prompt: str):
    response = openai.Image.create_edit(
        image=open(mask, "rb"),
        mask=open(mask, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    print('outpainted url', image_url)
    download_loc = mask.rsplit('/', 1)[0] + '/outpainted.png'
    # Download the image from the URL and save it to the local path
    response = requests.get(image_url)
    with open(download_loc, 'wb') as file:
        file.write(response.content)
    print('outpainted download loc', download_loc)
    return download_loc

def crop(outpainted_image):
    img = Image.open(outpainted_image)
    width, height = img.size
    top = int(height / 6)
    new_height = int(height * 2 / 3)  # Keep the middle six-ninths of the image
    cropped_img = img.crop((0, top, width, top+new_height))
    crop_save_loc = outpainted_image.rsplit('/', 1)[0] + '/final.png'
    cropped_img.save(crop_save_loc)
    return crop_save_loc

def correct_aspect_ratio(input_image_local_path: str, prompt: str):
    print('starting image aspect ratio correction')
    input_image_local_path = str(input_image_local_path)
    mask = create_mask(input_image_local_path)
    print('created mask at', mask)

    outpainted_loc = outpaint(mask, prompt)
    print('finished outpainting at', outpainted_loc)

    final_loc = crop(outpainted_loc)
    print('finished cropping at', final_loc)

    cloudinaryUploadResult = cloudinary.uploader.upload(final_loc)
    print('uploaded final image to', cloudinaryUploadResult['url'])
    return cloudinaryUploadResult['url']

# correct_aspect_ratio('/var/folders/vb/j4ndg33n0wx40znrr0rr4p2h0000gn/T/tmpwhda_yw5/0.jpeg',
#          'The Verdant Scale Dragon is a towering beast with shimmering green scales. She roars in defiance, an unfathomable energy swirling around her, deflecting spells originating from the planeswalkers.. In the style of high quality epic fantasy digital art')
