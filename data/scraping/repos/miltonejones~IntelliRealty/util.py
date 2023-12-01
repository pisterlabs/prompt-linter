
import os 
import json
import base64
import streamlit as st
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from geocode import get_lat_lon_from_address  

def convert_file_name(file_name): 
  file_name = file_name.split(".")[0] 
  words = file_name.split("_") 
  words = [word.capitalize() for word in words] 
  return " ".join(words) 


def get_document_list(city):
  folder_path = './src'
  file_object = {}

  # Iterate over each file in the folder
  for file_name in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file_name)):
      # Normalize the file name
      normalized_key = convert_file_name(file_name)
      basename = os.path.splitext(os.path.basename(file_name))[0]
      json_path = f'./json/{basename}.json'

      if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        if city is None or (json_data.get('city') and json_data['city'].lower() == city.lower()):
          # Create the object with properties path and favorite
          file_object[normalized_key] = {
              'path': './src/' + file_name,
              'favorite': json_data.get('favorite', False)  # Default to False if 'favorite' key doesn't exist
          }
      else:
        file_object[normalized_key] = {
            'path': './src/' + file_name,
            'favorite': False  # Default to False if JSON file doesn't exist
        }

  # Sort the file list alphabetically
  sorted_files = dict(sorted(file_object.items())) 
  return sorted_files
 

def save_json_to_file(filename, json_string, refresh=False):
  try:
      # Parse the JSON string to ensure it is valid JSON
      json_data = json.loads(json_string)

      # Get the basename and extension of the original filename
      basename, _ = os.path.splitext(os.path.basename(filename))

      # Create a new filename using the same basename and ".json" extension
      new_filename = f"./json/{basename}.json"

      # Check if the new filename already exists
      if os.path.exists(new_filename) and not refresh:
          print(f"Error: '{new_filename}' already exists. File not saved.")
      else:
        # Save the JSON data to the new file

        with open(new_filename, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f"JSON data saved to '{new_filename}'.")
  except json.JSONDecodeError as e:
      print(f"Error: Invalid JSON string - {e}")
  except Exception as e:
      print(f"Error: {e}")
 

def load_source_documents(city):
  folder_path = './src'
  sources = []
  embeddings = OpenAIEmbeddings()
  # Iterate over each file in the folder
  for file_name in os.listdir(folder_path):
    
    print ('file_name', file_name)
    # if os.path.isfile(os.path.join(folder_path, file_name)):
      
    folder_name = os.path.splitext(os.path.basename(file_name))[0]

    json_path = f'./json/{folder_name}.json'
    print ('json_path', json_path)

    if os.path.exists(json_path):
      with open(json_path, 'r') as file:
          json_data = json.load(file)

      if city is None or (json_data.get('city') and json_data['city'].lower() == city.lower()):
        folder_path = f'db/{folder_name}'
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
          print (folder_path)
          # Folder already exists, do something if needed 
          db = FAISS.load_local(folder_path, embeddings) 
          sources.append(db)

  first_element = sources[0]
  for element in sources[1:]:
      first_element.merge_from(element)
  print('sources', first_element)
  return first_element



@st.cache_data
def load_description(_qa, pdf, refresh=False):
  st.session_state.chat_history = []
 
  # Get the basename and extension of the original filename
  basename, _ = os.path.splitext(os.path.basename(pdf))

  # Create a new filename using the same basename and ".json" extension
  new_filename = f"./json/{basename}.json"

  if os.path.exists(new_filename) and not refresh:
    # Read and return the contents of the existing file
    with open(new_filename, 'r') as file:
      existing_json_string = file.read()
      print(f"File '{new_filename}' already exists. Returning its contents.")
      return {
        "answer": existing_json_string
      }
  else:
    result = _qa({"question": """
    create a JSON objects of properties about this listing. 
    use this interface.
    interface Property {
      title: string;
      city: string; // city only without state
      url: string;
      address: string;
      description: string; // one paragrah, 100 words or less. focus on location and amenities
      bedroomCount: number;
      bathroomCount: number;
      size: number;
      sizeType: string; sqft|sqm
      importantInformation: {
        viewings: string;
        externalStorage: string;
        garage: string;
        deliveryDate: string;
      };
      rentalDetails: {
        rentPrice: string;
        securityDeposit: string;
        rentalAgreement: string; 
        availableSince: string;
        status: string;
        acceptance: string;
        rentOptions: Array<{ // if more than one floor plan is listed, specify them here
          size: number;
          bedroomCount: number;
          bathroomCount: number;
          price: string;
          availableWhen: string;
        }>;
      };
      neighbourhood: string;
    }


    translate all information to english         
      """, "chat_history": []})
    
    data = result["answer"]

    answer = get_coords(data)

    save_json_to_file(pdf, answer, refresh)

    return result  

def get_coords(data):
  obj = json.loads(data) 
  
  info = get_lat_lon_from_address(obj["address"] + " " + obj["city"] + ", NL")
  print('geocode', info)
  obj["lat"] = info["lat"]
  obj["lon"] = info["lon"]
  return json.dumps(obj)



@st.cache_data
def load_source_document(pdf_file): 
  base_name = os.path.basename(pdf_file)
  folder_name = os.path.splitext(base_name)[0]
  embeddings = OpenAIEmbeddings()

  with st.spinner(text="Loading "+ folder_name +"..."): 
    folder_path = f'db/{folder_name}'
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
      # Folder already exists, do something if needed
      return FAISS.load_local(folder_path, embeddings) 
    else: 
      loader = PyPDFLoader(pdf_file)
      chunks = loader.load_and_split()  
      db = FAISS.from_documents(chunks, embeddings)  
      db.save_local(folder_path)
      return db
    


def save_uploaded_file(uploaded_file):
  # Create the 'src' folder if it doesn't exist
  if not os.path.exists('src'):
    os.makedirs('src')
  
  # Save the uploaded file to the 'src' folder
  file_path = os.path.join('src', uploaded_file.name)
  with open(file_path, 'wb') as f:
    f.write(uploaded_file.getbuffer())
  
  load_source_document(file_path)
  return file_path

def initialize_session():
  session_keys = {
    "chat_history": [],
    "selected_option": '',
    "city": "Amsterdam",
    "selected_desc": None,
    "llm_temparature": 0.7,
    "pdf_file": './src/Appartement te huur_ Entrepotbrug 151 1019 JH Amsterdam [funda].pdf' 
  }
 
  for key, value in session_keys.items():
    if key not in st.session_state:
      st.session_state[key] = value



def truncate_string(input_string, limit=25):
    item = st.session_state.source_docs[input_string]
    prefix = '❤️ ' if item['favorite'] else ''

    # Remove "Appartement te huur" from the string
    input_string = input_string.replace("Appartement te huur", "") 
    input_string = input_string.replace("Amsterdam", "") 
 
    # Truncate the string to the first and last 10 characters
    truncated_string = prefix + input_string[:limit] + '...' + input_string[-5:]

    return truncated_string
 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)