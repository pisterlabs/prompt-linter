from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
from flask import Flask, jsonify, request, render_template
import os
import validators
import requests

app = Flask(__name__)

# Set the static folder for Flask to serve static files from
app.static_folder = 'templates/static'

list_image_urls = []
index = None

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@app.route('/')
def index():
    """
    Render the index.html template for the homepage.
    """
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    Handle the image caption generation request.
    """
    global index

    if not index:
        return jsonify({'error': 'Please upload an image before generating a caption.'}), 400

    image_url = request.form.get('image_url')

    if not validate_image_url(image_url):
        return jsonify({'error': 'Invalid image URL.'}), 400

    if not is_valid_image(image_url):
        return jsonify({'error': 'The URL does not point to a valid image.'}), 400

    try:
        # Load the image and generate captions
        loader = ImageCaptionLoader(path_images=[image_url])
        list_docs = loader.load()
        index = VectorstoreIndexCreator().from_loaders([loader])
        result = index.query('Describe this image in 2 sentences.') #this is the instruction you are sending to openai
        return jsonify({'caption': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def validate_image_url(image_url):
    """
    Validate the format of the image URL.
    """
    if not validators.url(image_url):
        return False
    return True

def is_valid_image(image_url):
    """
    Check if the URL points to a valid image.
    """
    response = requests.head(image_url)
    content_type = response.headers.get('content-type')
    if content_type and 'image' in content_type:
        return True
    return False

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
