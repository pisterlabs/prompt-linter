import openai
import os
import webbrowser
openai.api_key = os.getenv('OPENAI_KEY')



# completion = openai.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "hi"}
#     ]
# )
# response_text = completion.choices[0].message.content
# print(response_text)


response = openai.images.generate(
    model="dall-e-3",
    prompt="Create a night-time scene of a boy, focused and coding on his laptop at his desk. Only depict the right side profile of the boy. The desk should be furnished with a steaming cup of coffee. Please ensure to include fine details in the image like the glowing laptop screen, and the chair upon which the boy is seated.",
    size="1024x1024",
    quality="standard",
    n=1,
)
print(response)
image_url = response.data[0].url
# print(image_url)
webbrowser.open(image_url)

# def generate(text):
#     response = openai.images.generate(
#         model="dall-e-3",
#         prompt=text,
#         size="1024x1024",
#         quality="standard",
#         n=1,
#     )
#     # returning the URL of one image as
#     # we are generating only one image
#     return response.data[0].url


# print(generate("a white siamese cat"))
# client = OpenAI()

# response = client.images.edit(
#     model="dall-e-2",
#     image=open("sunlit_lounge.png", "rb"),
#     mask=open("mask.png", "rb"),
#     prompt="A sunlit indoor lounge area with a pool containing a flamingo",
#     n=1,
#     size="1024x1024"
# )
# image_url = response.data[0].url


