import openai
from config import openaiapi
import main
import os

openai.api_key = openaiapi


def get_points(text):
    """Returns a list of keypoints from text."""
    extraction = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant working for the government, who reads the text given from a government file, and then creates an executive summary of the subject matter in the form of a bulleted list",
            },
            {
                "role": "user",
                "content": f"Please give an executive summary of the follwing text, making sure not to miss any major event or relevant information that may be useful for understanding the subject matter. Here is the text: '{text}'",
            },
        ],
    )
    return extraction.choices[0].message["content"].strip()


def keypoints(chunks):
    """Returns a list of keypoints from a list of chunks."""
    points = []
    for chunk in chunks:
        print(f"Getting keypoints for chunk: {chunk}")
        points.append(get_points(chunk))
        print(f"keypoints: {points}")
    return points


def kp(file_path):
    chunks = main.get_chunks(file_path)
    kp = keypoints(chunks)
    new_file_path = (
        f"./keypoints/{os.path.splitext(os.path.basename(file_path))[0]}.txt"
    )
    with open(new_file_path, "w") as file:
        file.write("\n".join(kp))
    return kp


file_path = "extracted_texts/Draft 6th PMC Minutes - Dairy Value Chain.txt"
kps = kp(file_path)
print(kps)
