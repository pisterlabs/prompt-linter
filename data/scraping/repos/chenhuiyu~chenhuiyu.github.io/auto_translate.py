import os
import re
import openai  # Make sure to install the openai package
from shutil import copyfile
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Initialize OpenAI API key (replace with your own key)
openai.organization = "org-vKWF67g2KjEjw9xurInSmRjF"
openai.api_key = "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"


# Function to process a directory and its subdirectories
def process_directory(directory_path):
    print(f"Exploring: {directory_path}")
    for root, dirs, files in os.walk(directory_path):
        print(f"Root: {root}")
        print(f"Dirs: {dirs}")
        print(f"Files: {files}")
        for file in files:
            if file.endswith(".md") and not(file.endswith('.zh-CN.md') or file.endswith('.en.md')):
                process_md_file(root, file)

@retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(20))
def translate_text(text):
    paragraphs = text.split("\n\n")
    translated_paragraphs = []

    for paragraph in paragraphs:
        # Extract asset_img tags
        asset_tags = re.findall(r"{% asset_img .*? %}", paragraph)
        remaining_text = re.sub(r"{% asset_img .*? %}", "", paragraph).strip()

        # Translate remaining text if any
        translated_text = remaining_text
        if remaining_text:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Please translate the Chinese text to English, maintaining the origin markdown format. Make sure your output is all in English. Do not make up any content or change the meaning of the original text.",
                    },
                    {"role": "user", "content": remaining_text},
                ],
                temperature=0.99,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            translated_text = response.choices[0]['message']['content'].strip()

        # Combine asset tags and translated text
        combined_text = " ".join(asset_tags) + " " + translated_text
        translated_paragraphs.append(combined_text.strip())

    return "\n\n".join(translated_paragraphs)


# Function to process each .md file
def process_md_file(root, file):
    # Skip if the file is already a translated or renamed version
    if file.endswith('.zh-CN.md') or file.endswith('.en.md'):
        return
    # Generate the new paths
    filename_without_ext = os.path.splitext(file)[0]
    renamed_filename = f"{filename_without_ext}.zh-CN.md"
    translated_filename = f"{filename_without_ext}.en.md"

    original_path = os.path.join(root, file)
    renamed_path = os.path.join(root, renamed_filename)
    translated_path = os.path.join(root, translated_filename)

    # Rename the file (by copying to a new name to keep the original as is)
    copyfile(original_path, renamed_path)

    # Read the original file content
    with open(original_path, "r", encoding="utf-8") as f:
        chinese_content = f.read()

    # Translate the content using GPT-4 API
    # Get the translated text
    translated_content = translate_text(chinese_content)

    # Write the translated content to the new file
    with open(translated_path, "w", encoding="utf-8") as f:
        f.write(translated_content)


# Start the process
if __name__ == "__main__":
    # Replace this with the path to your _posts directory
    process_directory("/Users/huiyu.chen/Documents/PersonalFiles/Blog/source/_posts")
