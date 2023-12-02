import requests
import openai
import os
import io

from tempfile import NamedTemporaryFile




# Imports the Google Cloud client library
from google.cloud import vision, videointelligence

GOOGLE_MAPS_API_KEY = os.environ['GOOGLE_MAPS_API_KEY']
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '.credentials/erp-2017-d844ca1fcc74.json'
openai.api_key = os.environ['OPENAI_KEY']


def audio_to_text(url):

    # Dapatkan ekstensi file dari URL
    file_extension = os.path.splitext(url)[1]
    file_extension = file_extension.split(";")[0].strip()

    response = requests.get(url)
    print(f'Url : {url}')

    # Buat file sementara dengan ekstensi yang sesuai
    with NamedTemporaryFile(suffix=file_extension, delete=True) as temp_file:

        temp_file.write(response.content)
        temp_file.flush()  # Pastikan semua data telah ditulis ke file
        
        print(f"Temporarily saved audio file to {temp_file.name}")


        try:
            # Gunakan file sementara untuk transkripsi
            temp_file.seek(0)  # Kembali ke awal file
            transcript = openai.Audio.transcribe("whisper-1", temp_file)
            print(f"Transkripsi: {str(transcript)}")
            text = transcript["text"]
        except Exception as e:
            print(f"Error (audio_to_text): {e}")
            text = f"Error: {e}"

    return text



def get_location_from_message(message):

    # Memisahkan latitude dan longitude dari pesan
    data_message = message.split('#')
    loc_name_1 = data_message[0]
    loc_name_2 = data_message[1]
    latitude = data_message[2]
    longitude = data_message[3]

    # loc_name, latitude, longitude = message.split('#')



    location = f'Location Name: {loc_name_1}, {loc_name_2} Latitude: {latitude}, Longitude: {longitude}'

    # GOOGLE_MAPS_API_KEY = os.environ['GOOGLE_MAPS_API_KEY']
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    endpoint = f"{base_url}?latlng={latitude},{longitude}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(endpoint)
    data = response.json()

    
    if data['status'] == 'OK':
        location += f" {data['results'][0]['formatted_address']}"
        return location
    else:
        return "Tidak dapat mendapatkan lokasi"




def recognize_image_from_url(url):
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = url

    #Get Label
    response = client.label_detection(image=image)
    labels = response.label_annotations
    detected_objects = [label.description for label in labels]

    image_info = f"Konteks Gambar: [{', '.join(detected_objects)}],"

    #Get Text from OCR
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if len(texts) > 0:    
        detected_texts = texts[0].description
        image_info += f"Text dalam gambar: [{str(detected_texts)}]"

    image_info = f'[{image_info}]'
    print(image_info)
    

    return image_info


 



# def detect_labels():

 
#     # Instantiates a client
#     client = vision.ImageAnnotatorClient()
#     # The name of the image file to annotate
#     file_name = os.path.abspath('./wakeupcat.jpg')


#     print(file_name)
#     # Loads the image into memory
#     with io.open(file_name, 'rb') as image_file:
#         content = image_file.read()
#     image = vision.Image(content=content)
#     # Performs label detection on the image file
#     response = client.label_detection(image=image)
#     labels = response.label_annotations
#     print('Labels:')
#     for label in labels:
#         print(label.description)


# detect_labels()


def analyze_video(url):
    # client = videointelligence.VideoIntelligenceServiceClient()

    # features = [videointelligence.Feature.LABEL_DETECTION]

    # operation = client.annotate_video(
    #     request={"features": features, "input_uri": video_uri}
    # )

    # print("\nProcessing video for label annotations:")
    # result = operation.result(timeout=180)

    # # Process video/segment level label annotations
    # segment_labels = result.annotation_results[0].segment_label_annotations
    # for i, segment_label in enumerate(segment_labels):
    #     print(f"Video label description: {segment_label.entity.description}")
    #     for category_entity in segment_label.category_entities:
    #         print(f"  Category: {category_entity.description}")

    # # Anda dapat menambahkan lebih banyak analisis sesuai kebutuhan Anda

    segment_labels = "Saat ini masih belum bisa mengenali video."

    return segment_labels





# video_url = "https://pati.wablas.com/video/GYDB-6EAA9B9407845120967D145FD3D05BCE.mp4"



# Di dalam kode Anda:
# if messageType == "video":
# video_description = analyze_video(video_url)
# message = f"_Video Description: {video_description}_\n\n"
# print(message)

#Pengecekan File Dokument
def analyze_document(url):
    text = "Saat ini masih belum bisa mengenali dokumen."
    return text