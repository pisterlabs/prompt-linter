from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from text_extraction import get_keyphrase_from_gpt
import numpy as np
import torch
from torchvision.ops import box_convert
import cv2
import base64
import random
import openai
from dotenv import load_dotenv
import time

load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")
CORS(app)

from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv

DENOMINATOR = 1000
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = "weights/" + WEIGHTS_NAME
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

CATCH_THRESHOLD = 0.5
SAVE_CROPPED = False

def save_cropped_image(cropped_img: np.ndarray, filename: str) -> str:
    path = os.path.join('cropped_images', filename)
    cv2.imwrite(path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    return path

def crop_image(image_source: np.ndarray, boxes: torch.Tensor, padding: int = 10) -> np.ndarray:
    h, w, _ = image_source.shape

    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    x1, y1, x2, y2 = map(int, xyxy[0])
    
    # Apply padding
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, w)
    y2 = min(y2 + padding, h)
    
    cropped_image = image_source[y1:y2, x1:x2]
    
    return cropped_image

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Get the image and the audio transcription from the request
        image_file = request.files['image']
        transcription = request.form['transcription']
        print(f'transcription: {transcription}, dtype: {type(transcription)}')
        item_name = get_keyphrase_from_gpt(transcription)
        if not item_name:
            item_name = transcription
        print(f'=== searching for: {item_name} ===')

        image_source, image = load_image(image_file)

        # Model prediction
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25
        print('=== prediciting ===')
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=item_name,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        success = False
        chances = 0
        if len(logits) > 0:
            print('logits: ', logits.tolist())
            max_confidence_index = torch.argmax(logits).item()
            selected_box = boxes[max_confidence_index]
            selected_logit = logits[max_confidence_index]

            probability = selected_logit.item()
            chances = int(probability * DENOMINATOR)

            success = (random.random() < probability) and (probability > CATCH_THRESHOLD)
            print('Catch success: ', success)

        if success:
            cropped_img = crop_image(image_source, selected_box.unsqueeze(0))
            cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_90_CLOCKWISE)
            if SAVE_CROPPED:
                filename = f"cropped_{int(time.time())}.jpeg"
                saved_path = save_cropped_image(cropped_img, filename)
                print(f'Cropped image saved at: {saved_path}')

            _, buffer = cv2.imencode('.jpeg', cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
            cropped_img_str = base64.b64encode(buffer).decode('utf-8')

            response = {
                'chances': chances,
                'success': success,
                'item_name': item_name,
                'cropped_image': cropped_img_str,
            }
        else:
            response = {
                'chances': chances,
                'success': success,
                'item_name': item_name,
                'cropped_image': None,
            }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)