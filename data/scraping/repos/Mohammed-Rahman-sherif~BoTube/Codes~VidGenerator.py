from moviepy.editor import ImageSequenceClip
import openai

openai.api_key = "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"
prompt = 'generate a sequence of images of a cat walking'

# Generate the images
response = openai.Image.create(prompt=prompt, n=10, size='1024x1024')

# Create a list of image paths
image_paths = [image['url'] for image in response['data']]

# Define the frames per second (fps)
fps = 1

# Create the video using the image paths and fps
clip = ImageSequenceClip(image_paths, fps=fps)

# Write the video to file
clip.write_videofile("output.mp4")
