import os
import openai
import time
from dotenv import load_dotenv
import tkinter as tk
from pywebcopy import save_webpage
import replicate
from tqdm import tqdm

load_dotenv()

# Download HTML

def download_html():
    global url
    url = url_entry.get()
    global project_name
    project_name = "gov_accessible_site"
    if url:
        try:
            save_webpage(
                url=url,
                project_folder=os.path.join(os.path.expanduser("~"), "Downloads"),
                project_name=project_name,
                bypass_robots=True,
                debug=True,
                open_in_browser=True,
                delay=None,
                threaded=False,
            )
            print("HTML source code downloaded successfully!")
        except Exception as e:
            print("An error occurred while downloading the HTML source code:", str(e))
    else:
        print("Please enter a valid URL.")
    root.destroy()

root = tk.Tk()
root.title("Website HTML Downloader")

url_label = tk.Label(root, text="Enter URL:")
url_label.pack()

url_entry = tk.Entry(root)
url_entry.pack()

enter_button = tk.Button(root, text="Enter", command=download_html)
enter_button.pack()

root.mainloop()

# Update Website

api_key = os.getenv('OPENAI_KEY')
openai.api_key = api_key

url = url.replace('~', '')
split_url = url.split('/')
recombined_url = '/'.join(split_url[2:-1])

def get_html_file_name(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Find the HTML file in the folder
    html_file_name = None
    for file in files:
        if file.endswith(".html"):
            html_file_name = file
            break

    return html_file_name

def get_images(folder_path):
    captions = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('gif'):
                recombined_file = os.path.join(root, file)
                output = replicate.run(
                    "rmokady/clip_prefix_caption:9a34a6339872a03f45236f114321fb51fc7aa8269d38ae0ce5334969981e4cd8",
                    input={"image": open(recombined_file, "rb")}
                )
                captions[file] = output

    return captions

html_file_name = get_html_file_name(os.path.join(os.path.expanduser("~"), "Downloads", project_name, recombined_url))

filename = os.path.join(os.path.expanduser("~"), "Downloads", project_name, recombined_url, html_file_name)
# filename = os.path.join(os.path.expanduser("~"), "Downloads", "website-short.html")
with open(filename, 'r', encoding='utf-8') as file:
    try:
        html_content = file.readlines()
    except UnicodeDecodeError as e:
        print("Unwanted character: ", e)

# Image Captioning

directory = os.path.join(os.path.expanduser("~"), "Downloads", project_name, recombined_url)

captions = get_images(directory)

print(captions)

prompt = '''Modify the following HTML code to be more accessible for the visually impaired, by making the font sizes bigger, fixing spacing issues by 
adding more space between visual elements. Also increase the contrast of images.
You will also replace the HTML tags "img" with code so I can hover over images to show text. Each of these HTML "img" tags will contain a "src" that has a 
file, and you will look for the corresponding text in a dictionary I will provide. I will provide the dictionary and 
HTMl code now. Return just the revised html code.'''

start_time = time.time()

print("Starting query...")

revised_html_parts = []
current_part = ""
character_count = 0
max_character_count = 7000

for line in html_content:
    line = line.strip()
    line_length = len(line)

    # Check if adding the line will exceed the maximum character count
    if character_count + line_length > max_character_count:
        revised_html_parts.append(current_part)
        current_part = line
        character_count = line_length
    else:
        current_part += line
        character_count += line_length

# Append the last part
if current_part:
    revised_html_parts.append(current_part)

# Process each part with the OpenAI API
revised_html = ""


for part in tqdm(revised_html_parts):
    response = openai.ChatCompletion.create(
        model = 'gpt-4', 
        messages=[{"role": "user", "content": prompt + "Dictionary: " + str(captions) +  ". HTML: " + part}],
    )
    revised_html += response.choices[0].message.content

end_time = time.time()
duration = end_time - start_time

print("Finished query...")

revised_filename = os.path.join(os.path.expanduser("~"), "Downloads", project_name, recombined_url, "revised-website.html")

with open(revised_filename, 'w', encoding='utf-8') as file:
    file.write(revised_html)

print("Revised HTML code saved successfully!")
print("Query duration:", duration, "seconds")

# API Test

# response = openai.ChatCompletion.create(
#     model = 'gpt-4', 
#     messages=[{"role": "user", "content": "Complete the text... france is famous for its"}]
#     )


# print(response.choices[0].message.content)