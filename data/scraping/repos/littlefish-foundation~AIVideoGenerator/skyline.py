import replicate
import openai
import mytoken
import os
import requests
import glob
import numpy as np
import pandas as pd
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
pd.set_option('display.max_colwidth', 80)  # default 50, the maximum width in characters of a column
pd.set_option('display.max_columns', 10)   # default 20, the maximum amount of columns in view 
pd.set_option('display.max_rows', 120)      # default 60, the maximum amount of rows in view
pd.set_option('display.expand_frame_repr', False)

def generate_dalle2_image(prompt, candidate_img_amount, filename, session_nr):
    # create/check a folders
    os.makedirs("candidate_images", exist_ok=True)
    response = openai.Image.create(prompt=prompt, n=candidate_img_amount, size="1024x1024")
    for idx, url in enumerate(response['data']):
        r = requests.get(url['url'], allow_redirects=True, stream=True)
        image = Image.open(r.raw)
        # save the original candidate image: image1_candidate1_1024px_original.png
        image.save("candidate_images/{}{}_candidate{}_{}px_original.png".format(filename, session_nr, idx+1, image.size[0]))
        print("candidate image file 'candidate_images/{}{}_candidate{}_{}px_original.png' written...".format(filename, session_nr, idx+1, image.size[0]))
    print("")
    return session_nr+1

def generate_candidate_maskedimages(prompt, candidate_img_amount, filename, session_nr, downscale_pct):
    # load and downscale the upscaled image below 1024px size according to user given %
    downscaled = Image.open("{}{}_{}px_upscaled.png".format(filename, session_nr-1, 4096)).resize((int(1024*(downscale_pct/100)), int(1024*(downscale_pct/100))))
    new_image = Image.new(mode="RGBA", size=(1024,1024))
    # you paste the original image (scaled down) on top of the new white image with full alpha layer on
    new_image.paste(downscaled, (int((1024/2)-(int(1024*(downscale_pct/100))/2)), int((1024/2)-(int(1024*(downscale_pct/100))/2))))
    new_image.save("{}{}_{}px_masked.png".format(filename, session_nr-1, 1024))
    print("masked image used for outpainting '{}{}_{}px_masked.png' written...".format(filename, session_nr-1, 1024))
    print("")
    # create/check a folders
    os.makedirs("candidate_images", exist_ok=True)
    response = openai.Image.create_edit(
        image=open("{}{}_{}px_masked.png".format(filename, session_nr-1, 1024), "rb"),
        mask=open("{}{}_{}px_masked.png".format(filename, session_nr-1, 1024), "rb"),
        prompt=prompt,
        n=candidate_img_amount,
        size="1024x1024")
    for idx, url in enumerate(response['data']):
        r = requests.get(url['url'], allow_redirects=True, stream=True)
        image = Image.open(r.raw)
        # save the original candidate image: image2_candidate1_1024px_original.png
        image.save("candidate_images/{}{}_candidate{}_{}px_original.png".format(filename, session_nr, idx+1, image.size[0]))
        print("candidate image file 'candidate_images/{}{}_candidate{}_{}px_original.png' written...".format(filename, session_nr, idx+1, image.size[0]))
    print("")
    return session_nr+1

def generate_prompt_from_image(filename, session_nr, prompt, prompt_df):
    # # not sure why, but to prevent the API from hanging indefenetly we refresh loading this token each time
    # load image to prompt pre-trained model
    replicateclient = replicate.Client(api_token=mytoken.REPLICATE_API_TOKEN)
    model = replicateclient.models.get("methexis-inc/img2prompt")
    # call the API and predict a result stored in output variable
    # this will take many seconds ~50sec to finish calculating
    print("Calculating image to prompt... 30 sec")
    output = model.predict(image=open("{}{}_{}px_origin.png".format(filename, session_nr-1, 1024), "rb"))
    #  removes any leading and trailing space characters
    result = output.strip()
    # store the original user input 'prompt' text as well as the image2prompt string to a text file: image1_prompt.txt
    open("{}{}_prompt.txt".format(filename, session_nr-1), 'w').write("User prompt:\n{}\nImage prompt:\n{}".format(prompt, result))
    print("image2prompt file '{}{}_prompt.txt' written...".format(filename, session_nr-1))
    # print the result to the console
    print("\tPrompt:\n", result, "\n")
    tmp = pd.DataFrame(data={'userPrompt': [prompt], 'imagePrompt': [result]})
    return pd.concat([prompt_df, tmp])

def enlarge_selected_image(filename, session_nr):
    ## upscale 
    # load/show image from disk (same image)
    print("Calculating image enlargement")
    # load original image
    image = Image.open("{}{}_{}px_origin.png".format(filename, session_nr-1, 1024))
    # upscale the image linearly
    image = image.resize((4096, 4096))
    # save the upscaled image: image1_4096px_upscaled.png
    image.save("{}{}_{}px_upscaled.png".format(filename, session_nr-1, image.size[0]))
    print("enlarged file for video '{}{}_{}px_upscaled.png' written...".format(filename, session_nr-1, image.size[0]))
    print("")

def generate_midjourney_image(prompt, candidate_img_amount, filename, session_nr):
    ### only to be used on the first image generation
    ### here we use Stable Diffusion MidJourney version 4
    # refresh load token id
    replicateclient = replicate.Client(api_token=mytoken.REPLICATE_API_TOKEN)
    # load the pre-trained model
    model = replicateclient.models.get("prompthero/openjourney")
    version = model.versions.get("9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb")
    # prepare extra words for the prompt to enable midjourney version 4
    prompt = f"mdjrny-v4 style {prompt}, highly detailed, masterpiece"
    # call API and generate results
    for output_nr in range(candidate_img_amount):
        response = version.predict(prompt=prompt,          # string of text
                                   width=768,              # Maximum size is 1024x768 or 768x1024 because of memory limits
                                   height=768,             # Maximum size is 1024x768 or 768x1024 because of memory limits
                                   num_outputs=1,          # Number of images to output (1..4)
                                   num_inference_steps=80, # High value, the better the image quality, because of more refinement steps
                                   guidance_scale=7)       # High value, the more closely it follows the prompt
        # download the image data from the url
        r = requests.get(response[0], allow_redirects=True, stream=True)
        # load the image into pillow for further processing
        image = Image.open(r.raw)
        # upscale the image linearly
        image = image.resize((1024, 1024))
        # create/check a folder
        os.makedirs("candidate_images", exist_ok=True)
        # save the original candidate image: image1_candidate1_1024px_original.png
        image.save("candidate_images/{}{}_candidate{}_{}px_original.png".format(filename, session_nr, output_nr+1, image.size[0]))
        print("candidate image file 'candidate_images/{}{}_candidate{}_{}px_original.png' written...".format(filename, session_nr, output_nr+1, image.size[0]))
    print("")
    return session_nr+1
 
# edit mytoken.py with your API TOKEN
# example: OPENAI_API_TOKEN = "f77b55967be209bc63a12038af9c09e0d3211996"
openai.api_key = mytoken.OPENAI_API_TOKEN

session_nr = 1
prompt_df = pd.DataFrame()

filename = str(input("Enter the 'filename' to save images to (ex: image): "))
candidate_img_amount = int(input("Enter the amount of candidate images to choose from (1..9) (ex: 4): "))
print("")

# ask the user for the fixed value 'downscale' percentage
downscale_pct = int(input("Enter downscale percentage(%), work best >50 (ex: 66): "))

# give the user a choice to either start with their own image file or generate the first with AI
startimage = str(input("If you want to start with prompt (press Enter), if you want to start with image (ex: image01.png): "))
# check if an empty string was given, meaning to go for a prompt
if len(startimage) == 0:
    print(f"You chose to ask for a prompt")
    mj_or_dalle = str(input("Enter the preferred image generator model (mj = midjourney) (dalle = DALLE-2): "))
    prompt = str(input("Enter an image prompt (ex: a white siamese cat): "))
    # generate and store candidate AI images using the prompt
    if mj_or_dalle == 'mj':
        session_nr = generate_midjourney_image(prompt, candidate_img_amount, filename, session_nr)
    else:
        session_nr = generate_dalle2_image(prompt, candidate_img_amount, filename, session_nr)
    while True:
        selection = int(input("Enter which image candidate you want to keep (0=retry, 1..{}) (ex: 1): ".format(candidate_img_amount)))
        if selection == 0:
            print("you chose to retry")
            prompt = str(input("Enter a new image prompt (ex: {}): ".format(prompt)))
            if mj_or_dalle == 'mj':
                session_nr = generate_midjourney_image(prompt, candidate_img_amount, filename, session_nr)
            else:
                session_nr = generate_dalle2_image(prompt, candidate_img_amount, filename, session_nr)
        else:
            break
    image = Image.open("candidate_images/{}{}_candidate{}_1024px_original.png".format(filename, session_nr-1, selection))
    print("candidate image file you chose to select 'candidate_images/{}{}_candidate{}_1024px_original.png'".format(filename, session_nr-1, selection))
    print("")
else:
    print(f"You have given a filename, checking if available...")
    while True:
        if os.path.isfile(startimage):
            # when the filename is found, then confirm
            print(f"your image '{startimage}' is found")
            break
        else:
            print(f"filename {startimage} does not exist")
            startimage = str(input("Enter your correct image filename please: "))
            continue
    image = Image.open(startimage)
    prompt = ''

# save the original image, include size: image1_1024px_origin.png
image.save("{}{}_{}px_origin.png".format(filename, session_nr-1, image.size[0]))
print("original file '{}{}_{}px_origin.png' written...".format(filename, session_nr-1, image.size[0]))
print("")

# generate the prompt from the selected image
prompt_df = generate_prompt_from_image(filename, session_nr, prompt, prompt_df)

# enlarge the selected image
enlarge_selected_image(filename, session_nr)

while True:
    prompt = str(input("Enter a new image prompt (no input=stop) (ex: {}): ".format(prompt)))
    if len(prompt) == 0:
        # when no input is given, STOP
        print(f'you chose to stop, starting video creation phase...')
        break
    else:
        # input is given, continue execution
        # generate and store new candidate AI images using the prompt
        session_nr = generate_candidate_maskedimages(prompt, candidate_img_amount, filename, session_nr, downscale_pct)

        while True:
            selection = int(input("Enter which image candidate you want to keep (0=retry, 1..{}) (ex: 1): ".format(candidate_img_amount)))
            if selection == 0:
                # when input 0 is given, RETRY
                print("you chose to retry")
                prompt = str(input("Enter a new image prompt (no input=stop) (ex: {}): ".format(prompt)))
                session_nr = generate_candidate_maskedimages(prompt, candidate_img_amount, filename, session_nr-1, downscale_pct)
            else:
                break
        image = Image.open("candidate_images/{}{}_candidate{}_1024px_original.png".format(filename, session_nr-1, selection))
        print("candidate image file you chose to select 'candidate_images/{}{}_candidate{}_1024px_original.png'".format(filename, session_nr-1, selection))
        print("")

        # save the original image, include size: image2_1024px_origin.png
        image.save("{}{}_{}px_origin.png".format(filename, session_nr-1, image.size[0]))
        print("original file '{}{}_{}px_origin.png' written...".format(filename, session_nr-1, image.size[0]))
        print("")

        # generate the prompt from the selected image
        prompt_df = generate_prompt_from_image(filename, session_nr, prompt, prompt_df)

        # enlarge the selected image
        enlarge_selected_image(filename, session_nr)
        continue

### starting the video generation
def store_prompt_data(filename, prompt_df):
    # data based on https://github.com/pharmapsychotic/clip-interrogator
    # available "mediums"   (cyberpunk art)        : https://github.com/pharmapsychotic/clip-interrogator/raw/main/clip_interrogator/data/mediums.txt
    # available "artists"   (Vincent Lefevre)      : https://github.com/pharmapsychotic/clip-interrogator/raw/main/clip_interrogator/data/artists.txt
    # available "Trending"  (Artstation)           : 
    # available "movements" (retrofuturism)        : https://github.com/pharmapsychotic/clip-interrogator/raw/main/clip_interrogator/data/movements.txt
    # available "flavors"   (synthwave, cityscape) : https://github.com/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator/data/flavors.txt
    # transform, clean and reshape the dataset
    split = prompt_df['imagePrompt'].str.split(', ', expand=True)
    prompt_df = pd.concat([prompt_df, split], axis=1)
    prompt_df = prompt_df.drop(['imagePrompt'], axis=1)
    flavors = prompt_df.iloc[:, 5:].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    columns = prompt_df.iloc[:, 5:].columns
    prompt_df = prompt_df.drop(columns, axis=1)
    prompt_df['Flavors'] = flavors
    split = prompt_df[1].str.split(' by ', expand=True)
    split.columns = ['Medium', 'Artist']
    prompt_df = pd.concat([prompt_df, split], axis=1)
    prompt_df = prompt_df.drop([1], axis=1)
    prompt_df.columns = ['userPrompt', 'imagePrompt', 'Trending', 'Movement', 'Flavors', 'Medium', 'Artist']
    prompt_df = prompt_df[['userPrompt', 'imagePrompt', 'Medium', 'Artist', 'Trending', 'Movement', 'Flavors']]
    # store the prompt text data DataFrame, without the index, utf-8 encoding, delimiter ';'
    prompt_df.to_csv("{}_prompt.csv".format(filename), encoding='utf-8', sep=';', index=False)
    return prompt_df

# store and clean the prompt text data
prompt_df = store_prompt_data(filename, prompt_df)
# get all upscaled images and sort by numeric value
upscaled_image_files = glob.glob(filename+"*_upscaled.png")
upscaled_image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

def leftside(imgsize, width):
    # how to calculate the distance from the left
    value = round((imgsize-width)/2)
    if value == 0:
        return value+1
    else:
        return value

def topside(imgsize, width, ratio):
    # how to calculate the distance from the top
    return round((imgsize-width*ratio)/2)

def rightside(imgsize, width):
    # how to calculate the distance from the left to outer side
    return round((imgsize+width)/2)

def downside(imgsize, width, ratio):
    # how to calculate the distance from the top to outer side
    return round((imgsize+width*ratio)/2)

def create_coordinates(width, imgsize, ratio):
    # calculate based on width value all 4 coordinates of the cropped out square
    coordinates = []
    coordinates.append(leftside(imgsize, width)-1)
    coordinates.append(topside(imgsize, width, ratio)-1)
    coordinates.append(rightside(imgsize, width)-1)
    coordinates.append(downside(imgsize, width, ratio)-1)
    # create a tuple format for the coordinate values
    return (coordinates[0], coordinates[1], coordinates[2], coordinates[3])

def calculate_width_values(imgsize, width, nr_of_crops):
    stepsize = (imgsize-width)/(nr_of_crops-1)
    return [int(np.ceil(x)) for x in np.arange(width, imgsize+1, stepsize)]

# parameters HD video 60 fps used for images crop and scale
pref_width = int(input("Enter the 'width'(px) of the video (ex: 1280): "))
pref_height = int(input("Enter the 'height'(px) of the video (ex: 720): "))
ratio  = pref_height/pref_width
width  = int(4096*(downscale_pct/100))
height = int(width*ratio)
# fps = int(input("Enter the 'fps' of the video (ex: 30): "))
fps = 60
# seconds_image = int(input("Enter the duration(sec) of each AI image on the screen (ex: 3): "))
seconds_image = 2
imgsize = 4096
nr_of_crops = fps * seconds_image
image_folder = filename
video_name = str(input("Enter the video 'filename' with 'codec' of the video (ex: finalvideo.mp4): "))
speed = round(1/fps, 4)

# upscaled_image_files
for idx, upscaled_image in enumerate(upscaled_image_files):
    print("starting new batch of images")
    # load in the upscaled image one-at-a-time
    image = Image.open(upscaled_image)
    image_coordinates = []
    for w in calculate_width_values(imgsize, width, nr_of_crops):
        image_coordinates.append(create_coordinates(w, imgsize, ratio))
    # create/check a folder
    os.makedirs(image_folder, exist_ok=True)
    # create the images inside the folder
    for img_number in range(nr_of_crops):
        cropped_image = image.crop(image_coordinates[img_number])
        cropped_image.resize((pref_width, pref_height)).save(image_folder + "/movieImg_{}.png".format(img_number+(idx*nr_of_crops)))
        print("cropped and rescaled image 'movieImg_{}.png' written ({}x{}) - step {}/{}".format((img_number+(idx*nr_of_crops)), pref_width, pref_height, (img_number+1)+(nr_of_crops*idx), nr_of_crops*len(upscaled_image_files)))

# get all the generated images *.png from the <image/> folder, sorted by numeric value
image_files = glob.glob(image_folder + "/movieImg_*.png")
image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# fix transition point between 2 images by removing one cropped image at the beginning of the new image
# I keep the original cropped pictures, but just remove certain items in the list
droplist = [x for x in range(0, len(image_files)-1, nr_of_crops)][1:]
for index in sorted(droplist, reverse=True):
    del image_files[index]

def duration_values_generator(timeperimage, downscale_pct, image_files, idx):
    # calculate a List of the duration time for each picture in the video
    # based on the downscaled image percentage cutoff (ex 33%) the first 67% of the time get a short duration time
    # then the remaining pictures receive a slightly longer duration time, hence slowing down little by little to half
    # this is needed because the transition to a new image is a zoomed out new version thus doubling the panning out that needs to be compensated with slower time
    # in case of 3 images = 10 cropped images:[2 2 2 2 2 2 2 2 2 2] 9 cropped images:[4 4 4 4 4 4 3 3.33, 2.67 2] 9 cropped images:[4 4 4 4 4 4 3 3.33, 2.67 2]
    durations = []
    first = (len(image_files)+idx)/(idx+1)
    second = (len(image_files)-first)/idx
    durations.extend(np.around(np.linspace(start=timeperimage*1, stop=timeperimage, num=round(((100-downscale_pct)/100)*first), endpoint=False), 3).tolist())
    durations.extend(np.around(np.linspace(start=timeperimage*1, stop=timeperimage, num=round((downscale_pct/100)*first), endpoint=True), 3).tolist())
    for i in range(idx):
        durations.extend(np.around(np.linspace(start=timeperimage*2, stop=timeperimage, num=round(((100-downscale_pct)/100)*second), endpoint=False), 3).tolist())
        durations.extend(np.around(np.linspace(start=timeperimage*1, stop=timeperimage, num=round((downscale_pct/100)*second), endpoint=True), 3).tolist())
    # here we then return the timetable list of each cropped image waiting time according to this algorithm, taking downscale percentage value cutoff in mind
    return durations

# load all images into memory to build a movie, using the timetable (how long each image stay in frame)
# durations = duration_values_generator(speed, downscale_pct, image_files, idx)
durations = duration_values_generator(0.02, downscale_pct, image_files, idx)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(sequence=image_files, durations=durations)
# write the video file to disk
clip.write_videofile(video_name, fps=fps)

# print out the final prompt data
print("")
print(prompt_df[['userPrompt', 'imagePrompt', 'Medium', 'Artist', 'Flavors']].reset_index(drop=True))