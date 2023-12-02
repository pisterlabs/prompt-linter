from app.utils.generate_unique_id import generate_unique_id
from docx import Document
from io import BytesIO
from app.utils.document_processor import chunk_text
import json
from app.utils.document_processor import extract_text_from_stream
from app.utils.summarize_document import summarize_document
from app.utils.vector_database_retriever import add_text_to_chroma
from app.utils.document_retriever import upload_document_to_s3, delete_document_from_s3, get_metadata_from_s3, get_url_from_s3, extract_text_from_s3, call_lambda_function
from app.utils.diagram_parser import serialize_to_json, deserialize_from_json, parse_results, create_text_representation
import subprocess
import tempfile
import os
from PIL import Image
from openai import OpenAI
import numpy as np
import cv2
import base64   
import whisper
import torch


client = OpenAI()



def upload_file_handler(uploaded_file, llm, content_type, file_type):
    if file_type == 'text':
        return text_file_handler(uploaded_file, llm, content_type)
    elif file_type == 'diagram':
        return diagram_file_handler(uploaded_file, llm, content_type)
    elif file_type == 'video':
        return video_file_handler(uploaded_file, llm, content_type)
    else:
        return None, 400 # invalid file type

def video_file_handler(video_file, llm, content_type):
    if content_type not in ['video/mp4']:
        return 400
    # call whisper to get transcript and timestamps
    transcript, texts, time_stamps = whisper_transcribe(video_file)
    # generate unique id with Document Level Hash
    file_id = generate_unique_id(transcript, 'text')
    # add transcript, id, and timestamps to chroma
    add_text_to_chroma(texts, file_id, time_stamps)
    # generate summary
    summary = summarize_document(texts, llm)
    # add file to S3 bucket: trace-ai-knowledge-base? or trace-ai-knowledge-base-videos?
    metadata = {
        "name": video_file.filename,
        "summary": summary,
        "content_type": content_type,
        "file_type": "video"
    }
    video_file.seek(0)
    upload_document_to_s3(video_file, file_id, metadata, content_type, bucket='trace-ai-knowledge-base-documents')
    # FOR RAG
    # Create an in-memory bytes buffer
    text_file = BytesIO(transcript.encode('utf-8'))
    text_file.seek(0)
    metadata = {
        "name": video_file.filename,
        "summary": summary,
        "content_type": "text/plain",
        "file_type": "text"
    }
    upload_document_to_s3(text_file, file_id, content_type="text/plain", bucket='trace-ai-transcripts')
    return file_id, 200
    
    #https://platform.openai.com/docs/guides/speech-to-text
    # post process with gpt4 if needed for spelling errors
    # pydub to segment if needed in the future

    # then, update RAG handler to handle video files
    # then, update frontend to handle video files and RAG videos


def whisper_transcribe(video_file):
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(video_file.read())
        temp_file_path = tmp_file.name
    try:
        numpy_audio = whisper.load_audio(temp_file_path)
        result = model.transcribe(numpy_audio, fp16=False)  # Set fp16 to False since we're using CPU

        # print the recognized text
        transcript = result['text']
        segments = result['segments']
        # language = result['language']

        # time_stamps = [(segment['start'], segment['end']) for segment in segments]
        time_stamps = [segment['start'] for segment in segments]
        texts = [segment['text'] for segment in segments]
        return transcript, texts, time_stamps
    except Exception as e:
        print(e)
        return None, None
    
    




# 'text':
        
        # for each file type:
        # generate unique id with Document Level Hash
        # generate summary
        # add file to chroma
        # add file to S3 bucket: trace-ai-documents

def text_file_handler(text_file, llm, content_type):
    if content_type not in ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return 400
    elif content_type == 'text/plain':
        text, chunked_text = extract_txt_text(text_file)
    elif content_type == 'application/pdf':
        text, chunked_text = extract_pdf_text(text_file)
    elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text, chunked_text = extract_docx_text(text_file)
    
    # generate unique id with Document Level Hash
    file_id = generate_unique_id(text, 'text')
    # generate summary
    summary = summarize_document(chunked_text, llm)
    # add text to chroma OR summary to chroma
    add_text_to_chroma(chunked_text, file_id)
    # add file to S3 bucket: trace-ai-documents

    metadata = {
        "name": text_file.filename,
        "summary": summary,
        "content_type": content_type,
        "file_type": "text"
    }
    upload_document_to_s3(text_file, file_id, metadata, content_type, bucket='trace-ai-knowledge-base-documents')
    return file_id, 200

def extract_txt_text(txt_file):
    # Read the text directly from the text file
    text = txt_file.read().decode('utf-8')  # assuming the text file is encoded in utf-8
    txt_file.seek(0)
    chunked_text = chunk_text(text)
    return text, chunked_text

def extract_pdf_text(pdf_file):
    # extract text from text file
    buffer = BytesIO(pdf_file.read())
    pdf_file.seek(0)
    text = extract_text_from_stream(buffer)
    chunked_text = chunk_text(text)
    return text, chunked_text

def extract_docx_text(docx_file):
    # Load the DOCX file and extract text
    doc = Document(docx_file)
    text = " ".join([p.text for p in doc.paragraphs])
    chunked_text = chunk_text(text)
    return text, chunked_text

# 'diagram':
        # for each file type:
        # generate unique id with Document Level Hash
        # generate summary: call lambda function to get class counts, bounding boxes, and confidence scores
        # create descriptive text representation of diagram: based on class counts, bounding boxes, and confidence scores
        # add text representation to chroma
        # add file to S3 bucket: trace-ai-documents




def diagram_file_handler(diagram_file, llm, content_type):
    # if content_type not in ['application/pdf', 'image/jpeg', 'image/png']:
    if content_type not in ['image/jpeg', 'image/png']:
        return 400

    # Read the binary content of the file
    diagram_content = diagram_file.read()
    diagram_file.seek(0)  # Go back to the start of the file
    image = Image.open(diagram_file) 

    # Create an in-memory file-like object from the original content
    diagram_file_copy1 = BytesIO(diagram_content)
    diagram_file_copy2 = BytesIO(diagram_content)
    diagram_file_copy1.seek(0)
    diagram_file_copy2.seek(0)
    # Generate a unique ID for the PDF file or image
    if content_type == 'application/pdf':
        file_id = generate_unique_id(diagram_content, 'binary')
    else:
        file_id = generate_unique_id(diagram_content, 'image')

    # upload temporary file to S3: trace-ai-images/input-images
    upload_document_to_s3(diagram_file_copy1, file_id, content_type = content_type, bucket='trace-ai-images', prefix='input-images')

    # call lambda function to get class counts, bounding boxes, and confidence scores
    lambda_response = call_lambda_function(file_id)

    # get s3 url for image
    image_url = get_url_from_s3(file_id, bucket='trace-ai-images', prefix='input-images')

    # generate image summary
    # TODO: eventually include class counts, connections, spatial relationships, etc. to prompt
    
    prompt = f"what's in the image, {diagram_file.filename}? Be specific about symbols and connections. Be sure to include the file name in your response."
    image_summary = summarize_image(image_url, prompt)

    # get results for lambda function response
    results = json.loads(lambda_response['body']).get('results', None)

    # get class counts, bounding boxes, and confidence scores from results
    class_counts, bounding_boxes, confidence_scores = parse_results(results)

    # create descriptive text representation of diagram: based on class counts, bounding boxes, and confidence scores
    text_representation = create_text_representation(diagram_file.filename, class_counts, bounding_boxes, confidence_scores)

    # add file to chroma OR summary to chroma
    chunked_text = chunk_text(image_summary + text_representation)
    add_text_to_chroma(chunked_text, file_id)

    classification_data = serialize_to_json(diagram_file.filename, class_counts, bounding_boxes, confidence_scores, results, image_summary)

    metadata = {
        "name": diagram_file.filename,
        "summary": text_representation,
        "content_type": content_type,
        "file_type": "diagram",
    }

    # Convert the dictionary to a JSON string
    classification_data_string = json.dumps(classification_data)

    # Convert the JSON string to bytes
    classification_data_bytes = classification_data_string.encode('utf-8')

    # Create a file-like object from the JSON bytes
    classification_dataobj = BytesIO(classification_data_bytes)
    upload_document_to_s3(classification_dataobj, file_id, content_type='application/json', bucket='trace-ai-diagram-metadata')
    # # add file to S3 bucket: trace-ai-knowledge-base-documents
    upload_document_to_s3(diagram_file_copy2, file_id, metadata, content_type, bucket='trace-ai-knowledge-base-documents')

    # delete temporary file from S3: trace-ai-images/input-images
    delete_document_from_s3(file_id, bucket='trace-ai-images', prefix='input-images')
    # annotate image with bounding boxes and class labels 
    annotated_image = draw_bounding_boxes(image, results)

    img_byte_arr = BytesIO()
    annotated_image.save(img_byte_arr, format=content_type.split('/')[-1].upper())
    img_byte_arr.seek(0)
    # upload processed image to S3: trace-ai-images/processed-images
    upload_document_to_s3(img_byte_arr, file_id, content_type=content_type, bucket='trace-ai-images', prefix='processed-images')
    print(image_summary)
    return file_id, 200



def draw_bounding_boxes(image, results):
    cv_image = np.array(image)
    # Convert RGB to BGR for OpenCV
    cv_image = cv_image[:, :, ::-1].copy()
    predictions = results['predictions'][0]
    boxes = predictions['output_0']
    confidences = predictions['output_1']
    class_ids = predictions['output_2']
    labels = [str(int(class_id)) for class_id in class_ids]  # Convert class IDs to string labels; adjust as needed

    for box, confidence, label in zip(boxes, confidences, labels):
        if all(v == 0.0 for v in box):  # skip boxes with all zeros
            continue
        x1, y1, x2, y2 = map(int, [box[0] * cv_image.shape[1], box[1] * cv_image.shape[0], box[2] * cv_image.shape[1],
                                   box[3] * cv_image.shape[0]])
        label_with_confidence = f"{label} ({confidence:.2f})"
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cv_image, label_with_confidence, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(cv_image[:, :, ::-1])



def summarize_image(image_url, prompt="Describe the image in detail. Be specific about symbols and connections."):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content
