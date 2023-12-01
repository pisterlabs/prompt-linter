from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from .models import db, User, Manga
from . import app
from config import Config
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import base64, fitz, os
from PIL import Image
from io import BytesIO
from langchain.llms import OpenAI
from sqlalchemy import or_

api_key = os.getenv('OpenAI_KEY')

llm = OpenAI(openai_api_key = api_key, temperature=0.9)

app.config.from_object(Config)
jwt = JWTManager(app)

def analyze_image(image_content):
    client = vision_v1.ImageAnnotatorClient()

    image_byte_array = BytesIO()
    image_content.save(image_byte_array, format="PNG")
    image_bytes = image_byte_array.getvalue()

    image = types.Image(content=image_bytes)
    # Perform label detection
    response = client.label_detection(image=image)
    # Extract labels
    labels = [label.description for label in response.label_annotations]
    # Perform explicit content detection
    explicit_content = response.safe_search_annotation
    # Check for explicit content
    explicit_labels = [
        "UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY"
    ]
    is_explicit = explicit_content.adult in [explicit_labels.index("POSSIBLE"), explicit_labels.index("LIKELY"), explicit_labels.index("VERY_LIKELY")]

    return labels, is_explicit

def extract_images_from_pdf(base64_content):
    pdf_content = base64.b64decode(base64_content)
    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

    page_images = []

    for page_num in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_num]
        # Get the pixmap for the page
        pixmap = page.get_pixmap()
        # Get the image data as PNG
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

        page_images.append(img)

    pdf_document.close()

    return page_images

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    print(len(hashed_password))
    new_user = User(name=data['name'], email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()

    if user and check_password_hash(user.password, data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({'message': 'Invalid email or password'}), 401

# Manga Upload and Reading

@app.route('/api/upload', methods=['POST'])
@jwt_required()
def upload_manga():
    data = request.get_json()
    current_user = User.query.get(get_jwt_identity())

    managa_images = extract_images_from_pdf(base64_content=data['manga_content'])
    
    for image in managa_images:
        labels, is_explicit = analyze_image(image_content=image)
        if is_explicit:
            return jsonify({'message': 'Manga contains explicit content'}), 400
    
    existing_manga = Manga.query.filter_by(
        manga_name=data['manga_name'],
        manga_issue_no=['manga_issue_no']
    ).first()

    if existing_manga:
        return jsonify({'message': 'Manga with the same details already exists'}), 400
    
    new_manga = Manga(
        uploaded_by=current_user.id,
        manga_name=data['manga_name'],
        manga_author=data['manga_author'],
        manga_tags=data['manga_tags'],
        manga_issue_no=data['manga_issue_no'],
        manga_content=data['manga_content']
    )

    current_user.manga_count += 1
    db.session.add(new_manga)
    db.session.commit()
    
    return jsonify({'message': 'Manga uploaded successfully'}), 201

@app.route('/api/manga/<int:manga_id>', methods=['GET'])
def read_manga(manga_id):
    manga = Manga.query.get(manga_id)
    if manga:
        return jsonify({
            'manga_name': manga.manga_name,
            'manga_author': manga.manga_author,
            'manga_tags': manga.manga_tags,
            'manga_issue_no': manga.manga_issue_no,
            'uploaded_at': manga.uploaded_at.strftime("%Y-%m-%d %H:%M:%S")
        }), 200
    else:
        return jsonify({'message': 'Manga not found'}), 404

# Manga Search
@app.route('/api/search', methods=['POST'])
def search_manga():
    query = request.get_json()

    prompt_1 = "This is a user query for manga search in a website: "
    prompt_2 = " Search for manga keywords in it like author name or manga name and return your response identified keywords seperated by ;"
    prompt = prompt_1 + query['query'] + prompt_2
    keywords = llm(prompt).split(';')

    prompt_3 = " Search for manga tags in it like genre, shonen or other details and return your response identified tags seperated by ;"
    prompt = prompt_1 + query['query'] + prompt_3
    tags = llm(prompt).split(';')

    results = set()  # Use a set to avoid duplicates

    if keywords:
        keyword_conditions = [Manga.manga_name.ilike(f"%{keyword}%") | Manga.manga_author.ilike(f"%{keyword}%") for keyword in keywords]
        results.update(Manga.query.filter(or_(*keyword_conditions)).all())

    if tags:
        tag_conditions = [Manga.manga_tags.ilike(f"%{tag}%") for tag in tags]
        results.update(Manga.query.filter(or_(*tag_conditions)).all())

    if results:
        response = [{'manga_name': manga.manga_name, 'manga_author': manga.manga_author} for manga in results]
        return jsonify(response), 200
    else:
        return jsonify({'message': 'No matching manga found'}), 404

if __name__ == '__main__':
    db.init_app(app)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
