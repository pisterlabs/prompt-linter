from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app) 
client = OpenAI()

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    content = request.json
    prompt = content['prompt']
    
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1792"
        )
        # Assuming the response contains a direct URL to the image
        image_url = response.data[0].url
        return jsonify({'url': image_url})
    except client.error.OpenAIError as e:
        return jsonify({'error': str(e)}), e.http_status

if __name__ == '__main__':
    app.run(debug=True)