from flask import Flask, render_template, request, send_from_directory, session, send_from_directory
import os
import base64
from openai import OpenAI

app = Flask(__name__)

# Initialize the OpenAI client and set the API key
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'uploads'

# Model name
model = "gpt-4-vision-preview"

# This session key stores the dialog history
SESSION_KEY_DIALOG_HISTORY = 'dialog_history'

image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'working_image.jpg')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def reset_dialog_history():
    session[SESSION_KEY_DIALOG_HISTORY] = []

def generate_images(prompt, size, style, quality):
    model = "dall-e-3"  # Adjust the model name if needed
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        style=style,
        quality=quality,
    )

    image_url = response.data[0].url
    return image_url

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/vision")
def vision_index():
    reset_dialog_history()  # Reset history whenever index is loaded
    return render_template("vision_index.html")

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/generate", methods=["POST"])
def generate_description():
    image = request.files.get("image")
    description_text = request.form.get("description_text", "What's in this image?")

    if image and image.filename != "":
        # If a new file is selected, save it as working_image.jpg
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'working_image.jpg')
        image.save(image_path)
    else:
        # If no new file is selected, continue using the existing working_image.jpg
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'working_image.jpg')

    base64_image = encode_image(image_path)

        # Store only the filename in the session, not the entire image content
    session[SESSION_KEY_DIALOG_HISTORY].append({
        "role": "system",
        "content": "User uploaded a new image." if image and image.filename != "" else "User continued with the existing image."
    })
    session[SESSION_KEY_DIALOG_HISTORY].append({
        "role": "user",
        "content": [
            {"type": "text", "text": description_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ],
    })


    messages = session.get(SESSION_KEY_DIALOG_HISTORY, [])
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
    )
    
    # after the response, add bot's response to history
    session[SESSION_KEY_DIALOG_HISTORY].append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    
    description = response.choices[0].message.content
    return render_template("vision_index.html", description=description, history=messages, image_uploaded=os.path.exists(image_path))

@app.route("/reset", methods=["POST"])
def reset_session():
    reset_dialog_history()
    return render_template('vision_index.html', description='', history=[])

@app.route("/dalle", methods=["GET", "POST"])
def dalle_index():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        size = request.form.get("size")
        style = request.form.get("style")
        quality = request.form.get("quality")
        image_url = generate_images(prompt, size, style, quality)
        session["prompt"] = prompt
        session["size"] = size
        session["style"] = style
        session["quality"] = quality

        return render_template("dalle_index.html", prompt=prompt, size=size, style=style, quality=quality, image_url=image_url)
    else:
        prompt = session.get("prompt", "")
        size = session.get("size", "1024x1024")
        style = session.get("style", "vivid")
        quality = session.get("quality", "standard")
        return render_template("dalle_index.html", prompt=prompt)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
