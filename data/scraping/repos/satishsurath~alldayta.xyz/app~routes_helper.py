import hashlib
import os
import time
import openai
import random
from pdfminer.high_level import extract_text
from io import BytesIO
from app.forms import UploadSyllabus
from werkzeug.utils import secure_filename
from app import app
from flask import session
from app.file_operations import (
    read_from_file_json,
    read_from_file_text,
    check_folder_exists,
    list_folders,
    rename_folder,
    delete_folder,
    create_course_folder_with_metadata,
    allowed_file,
    delete_file,
    get_first_txt_file,
    get_file_path,
    delete_files_in_folder

)


# This function handles an uploaded PDF file from a form, extracts text from the PDF, generates a hash of the extracted text for naming, checks if a text file already exists for the course and deletes it if so, and finally saves the PDF and the extracted text to the designated locations.
def save_syllabus(form, course_name):
    # Get the uploaded PDF file
    syllabus_file = form.syllabus.data
    #generate file names and full file paths for the Syllabus file
    extension = os.path.splitext(syllabus_file.filename)[1]
    filename = secure_filename("Syllabus-" + course_name + extension)
    user_folder = session['folder']
    syllabus_path = get_file_path(app.config['FOLDER_UPLOAD'], user_folder, course_name, filename)
    #delete old syllabus before saving the new one if its present
    delete_previous_syllabus(course_name)
    #save the syllabus file
    syllabus_file.save(syllabus_path)

#function to check if there is another file that starts with <course_name> + "Syllabus" and delete it if it exists. input is couse_name
def delete_previous_syllabus(course_name):
    user_folder = session['folder']
    folder_path = os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name)
    #get a list file names in the course folder
    contents = os.listdir(folder_path)
    # Check if there is a file that starts with course_name + "Syllabus" in contents
    for file in contents:
        if file.startswith("Syllabus" + course_name):
            file_path = os.path.join(folder_path, file)
            try:
                os.remove(file_path)
                print(f"Deleted previous syllabus: {file_path}")
                app.logger.info(f"Deleted previous syllabus: {file_path}")
            except Exception as e:
                print(f"Error deleting the file {file_path}. Reason: {str(e)}")

# define a new function to read the contents of the syllabus file. input is the course_name
def read_syllabus(course_name):
    user_folder = session['folder']
    folder_path = os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name)
    #get the file name of the syllabus file
    contents = os.listdir(folder_path)
    # Check if there is a file that starts with course_name + "Syllabus" in contents
    for file in contents:
        if file.startswith(course_name + "Syllabus"):
            file_path = os.path.join(folder_path, file)
            # File existis!
            # Check if it has been processed into chunks
            



    


    previous_syllabus_file = get_first_txt_file(app.config['FOLDER_UPLOAD'], user_folder, course_name, course_name + "Syllabus")
    if previous_syllabus_file:
        #delete the file
        delete_file(app.config['FOLDER_UPLOAD'], user_folder, course_name, previous_syllabus_file)

# This function checks if the '-originaltext.csv' and '-originaltext.npy' versions of each file in the provided list exist in the 'Textchunks' and 'EmbeddedText' folders respectively, and returns their existence status.
def check_processed_files(contents, parent_folder):
    processed_files_info = []
    for file in contents:
        file_without_ext = os.path.splitext(file)[0]
        processed_csv_file_name = file_without_ext + "-originaltext.csv"
        processed_npy_file_name = file_without_ext + "-originaltext.npy"
        processed_csv_file_path = os.path.join(parent_folder, "Textchunks", processed_csv_file_name)
        processed_npy_file_path = os.path.join(parent_folder, "EmbeddedText", processed_npy_file_name)
        csv_exists = os.path.isfile(processed_csv_file_path)
        npy_exists = os.path.isfile(processed_npy_file_path)
        # Add the csv and npy existence check results to the list
        processed_files_info.append([file, csv_exists, npy_exists])
    return processed_files_info

# This function checks if 'textchunks.npy' and 'textchunks-originaltext.csv' files are present in a given course directory, and returns their status (present or not) and size.
def detect_final_data_files(course_name):
    file_info = {}
    for file_name in ['Textchunks.npy', 'Textchunks-originaltext.csv']:
        file_path = os.path.join(course_name, file_name)
        if os.path.isfile(file_path):
            file_info[file_name] = {
                'present': True,
                'name': file_name,
                'size': os.path.getsize(file_path)
            }
        else:
            file_info[file_name] = {
                'present': False,
                'name': file_name,
                'size': None
            }
    return file_info


# Define a retry decorator with exponential backoff
def retry_with_exponential_backoff(func):
    def wrapper(*args, **kwargs):
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds
        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except openai.error.RateLimitError as e:
                print("Rate limit exceeded. Retrying after delay...")
                time.sleep(retry_delay)
                # Increase the delay for the next retry with some random jitter
                retry_delay *= 2 * random.uniform(0.8, 1.2)
        # If max_retries exceeded, raise an exception
        raise Exception("API rate limit exceeded even after retries.")
    return wrapper


# read metadata from the course folder and store them in session variables
def load_course_metadata(metadata):
    # save the values of the metadata in session
    session['classname'] = metadata['classname']
    session['professor'] = metadata['professor']
    session['assistants'] = metadata['assistants']
    session['classdescription'] = metadata['classdescription']
    session['assistant_name'] = metadata['assistant_name']

def process_syllabus(course_name):
    app.logger.info(f"Entered process_syllabus")
    user_folder = session['folder']
    
    ####------ to be done
    # Assuming you have similar functions for syllabus processing
    #chop_syllabus_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
    #embed_syllabus_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
    
    try:
        ####------ to be done
        #create_final_data_syllabus_given_course_name(os.path.join(app.config['FOLDER_UPLOAD'], user_folder, course_name))
        app.logger.info(f"Completed process_syllabus for COURSE NAME: {course_name}")
        return True
    except Exception as e:
        app.logger.error(f"An error occurred while processing syllabus: {e}")
        return False