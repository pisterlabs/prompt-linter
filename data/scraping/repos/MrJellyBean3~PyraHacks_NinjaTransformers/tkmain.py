import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
import requests
import openai
import random
import time
#set openai api key and model
openai_credentials_file=open("SECRET.txt","r")
key=openai_credentials_file.readline().split(" ")[0]
openai_credentials_file.close()
openai_model="gpt-4"#gpt-3.5-turbo"
openai.api_key = key

# This function will get the astrology summary
# Replace this with actual call to GPT
def get_astrology_summary():
    choices=["a planet","a space phenomenon","a sun","a black hole", "a specific god and their relation to astrology"]
    choice=random.choice(choices)
    prompt_text="I want you to tell me about astrology, "+choice+". I want you to give me it in terms of <Topic>:<Description or Story>:<Image Prompt> example 'Connecting Stars and Personalities: Astrology analyzes the position of celest...: Stars with ancient personalities art' Make it short, simple, and make the topic specifi not something generic like 'Astrology'.\n"
    
    print(prompt_text)
    completion=openai.ChatCompletion.create(
            model=openai_model,
            messages=[{"role": "user", "content": str(prompt_text)}],
    )
    filtered_text=completion.choices[0].message.content
    print(filtered_text)
    return(filtered_text.split(":")[1],filtered_text.split(":")[2])
def get_astrology_image(image_prompt):
    response = openai.Image.create(
        prompt=image_prompt,
        n=1,
        size="512x512"#"1024x1024"#"512x512"
    )
    image_url = response['data'][0]['url']
    url = image_url
    response = requests.get(url)
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
    #read image.jpg
    image=Image.open("image.jpg")
    return image

class AstrologyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("400x400")

        # Set up labels
        self.text_label = tk.Label(self, text="", wraplength=300)
        self.text_label.pack(pady=10)

        # Placeholder image
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        # Next button
        self.next_button = tk.Button(self, text="Next", command=self.update_content)
        self.next_button.pack(pady=10)

        self.update_content()

    def update_content(self):
        summary,image_prompt = get_astrology_summary()
        image = get_astrology_image(image_prompt)

        # Update text
        self.text_label.configure(text=summary)

        # Update image
        image.thumbnail((200,200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # keep a reference!

if __name__ == "__main__":
    app = AstrologyApp()
    app.mainloop()
