import openai
import confidential
import requests
from PIL import Image, ImageFont, ImageDraw
import re
import cv2

openai.api_key = confidential.openai_api_key

def func(num, desc, title):
    description = desc
    
    blank_large = Image.new('RGBA', (1024, 1024), (255, 0, 0, 0))

    blank_large.save(confidential.Blank_Large, 'PNG')

    while True:
        try:
            response = openai.Image.create_edit(
                image=open(confidential.Blank_Large, "rb"),
                mask=open(confidential.Blank_Large, "rb"),
                prompt=description,
                n=1,
                size="1024x1024"
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    image_url = response['data'][0]['url']

    img_data = requests.get(image_url).content

    with open(confidential.Original_Image, 'bw') as handler:
            handler.write(img_data)

    # 1024 x 1024 image
    original = Image.open(confidential.Original_Image)

    # 700 top pixels original
    original_with_bottom = original.crop((0, 324, 1024, 1024))

    # adds 324 blank pixels on the bottom
    blank_large.paste(original_with_bottom, (0, 0))

    blank_large.save(confidential.Blank_Bottom, 'PNG')

    while True:
        try:
            response = openai.Image.create_edit(
                image=open(confidential.Blank_Bottom, "rb"),
                mask=open(confidential.Blank_Bottom, "rb"),
                prompt=description,
                n=1,
                size="1024x1024"
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    image_url = response['data'][0]['url']

    img_data = requests.get(image_url).content

    with open(confidential.Complete_Bottom, 'bw') as handler:
        handler.write(img_data)

    # 724 bottom pixels original
    original_with_top = original.crop((0, 0, 1024, 724))

    new_blank_large = Image.new('RGBA', (1024, 1024), (255, 0, 0, 0))

    # 300 new pixels on the top
    new_blank_large.paste(original_with_top, (0, 300))

    new_blank_large.save(confidential.Blank_Top, 'PNG')

    while True:
        try:
            response = openai.Image.create_edit(
                image=open(confidential.Blank_Top, "rb"),
                mask=open(confidential.Blank_Top, "rb"),
                prompt=description,
                n=1,
                size="1024x1024"
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    image_url = response['data'][0]['url']

    img_data = requests.get(image_url).content

    with open(confidential.Complete_Top, 'bw') as handler:
            handler.write(img_data)

    toptop = Image.open(confidential.Complete_Top)

    #300 pixels new (on the top) from 0 to 300
    top3 = toptop.crop((0, 0, 1024, 300))

    top3.save(confidential.Cropped_Top, 'PNG')

    bottombottom = Image.open(confidential.Complete_Bottom)

    # 324 new pixels (on the bottom) from 700 to 1024
    bottom2 = bottombottom.crop((0, 700, 1024, 1024))

    bottom2.save(confidential.Cropped_Bottom, 'PNG')

    im1 = cv2.imread(confidential.Cropped_Top)

    im2 = cv2.imread(confidential.Original_Image)

    im3 = cv2.vconcat([im1, im2])

    im4 = cv2.imread(confidential.Cropped_Bottom)

    im5 = cv2.vconcat([im3, im4])

    cv2.imwrite(confidential.Final_Image + str(num) + ".png", im5)

    final_image = Image.open(confidential.Final_Image + str(num) + ".png")

    final_image = final_image.crop((0, 0, 1000, 1600))

    text_size = 120
    
    fontColor = "white"
    shadowcolor = "black"

    while True:
        font = ImageFont.truetype("Arial Bold.ttf", text_size)
        message = title
        draw = ImageDraw.Draw(final_image)
        W, _ = final_image.size
        _, _, w, _ = draw.textbbox((0, 0), message, font=font)
        if w > .8 * W:
            text_size = text_size - 1
            continue
        break

    draw.text(((W-w)/2 - 1, 50 + 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2 + 1, 50 - 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2, 50 + 1, 50 +1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2 - 1, 50 - 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2, 50), message, font=font, fill=fontColor)

    text_size = 120

    while True:
        font = ImageFont.truetype("Arial Bold.ttf", text_size)
        message = 'Rhett Konner'
        draw = ImageDraw.Draw(final_image)
        W, _ = final_image.size
        _, _, w, _ = draw.textbbox((0, 0), message, font=font)
        if w > .8 * W:
            text_size = text_size - 1
            continue
        break

    draw.text(((W-w)/2 - 1, 1400 + 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2 + 1, 1400 - 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2 + 1, 1400 + 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2 - 1, 1400 - 1), message, font=font, fill=shadowcolor)
    draw.text(((W-w)/2, 1400), message, font=font, fill=fontColor)

    final_image.save(confidential.Final_Image_With_Words + str(num) + ".png", 'PNG')

def main():
    number_of_covers = input('How many covers would you like to make? ')
    
    text = 'Create a list of ' + number_of_covers + ' examples of sentences like this: they start with "An epic fantasy painting montage of" followed by three objects. For example, "An epic fantasy painting montage of a saucepan, a wine glass, and a bar of soap." or "An epic fantasy painting montage of a portapotty, a pair of high heels, and a coffee mug."'
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = [{"role" : "user", "content": text}],
                temperature = 0.8,
                max_tokens = 12000,
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.InvalidRequestError:
            print("Invalid Request Error (too many tokens)")
            most_tokens -= 100
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break
    
    script = response.choices[0].message["content"]
    
    i = 0
    if int(number_of_covers) == 1:
        func(i, script)
    for j in re.split("[0-9]+\.", script):
        if(len(j) > 1 and j.startswith(" An epic fantasy painting")):
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model = "gpt-3.5-turbo-16k",
                        messages = [{"role" : "user", "content": "Write a short title (preferably two words or less) for a novel that matches the following painting description: " + j}],
                        temperature = 0.8,
                        max_tokens = 12000,
                    )
                except openai.error.RateLimitError:
                    print("Rate Limit Error")
                    continue
                except openai.error.InvalidRequestError:
                    print("Invalid Request Error (too many tokens)")
                    most_tokens -= 100
                    continue
                except openai.error.APIError:
                    print("API Error: probably bad gateway")
                    continue
                except openai.error.ServiceUnavailableError:
                        print("openai.error.ServiceUnavailableError")
                        continue
                break
            title = response.choices[0].message["content"]
            if str(title)[0].isalpha() == False and str(title)[-1].isalpha() == False:
                title = title[1:-1]
            print(title)
            func(i, j, title)
            i += 1

main()