import re
import os
import json
import openai
import sqlite3
import win32com.client as win32
import win32com.client.gencache
from openai import OpenAI
from contextvars import ContextVar
from typing import Optional, Callable, List
from dotenv import load_dotenv
import datetime

# Load environment variables from .env.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client with your API key
openai.api_key = api_key

# Ensure the generation of COM libraries.
win32.gencache.EnsureDispatch("Outlook.Application")
constants = win32.constants
outlook = win32com.client.Dispatch("Outlook.Application")

# Type annotation for Cursor (assuming sqlite3, replace with your actual cursor type if different)
Cursor = sqlite3.Cursor

# Context variable for SQLite connection
matrix_connection_var: ContextVar[Optional[sqlite3.Connection]] = ContextVar(
    "matrix_connection", default=None
)

# Context variable for matrix cursor
matrix_cursor_var: ContextVar[Optional[sqlite3.Cursor]] = ContextVar(
    "matrix_cursor_var", default=None
)

import threading


class MatrixDatabaseContextManager:
    _lock = threading.Lock()
    _ref_count = 0

    def __enter__(self):
        with MatrixDatabaseContextManager._lock:
            MatrixDatabaseContextManager._ref_count += 1

            connection = matrix_connection_var.get(None)
            if connection is None:
                connection = sqlite3.connect("Matrix.db")
                matrix_connection_var.set(connection)

            self.cursor = connection.cursor()
            matrix_cursor_var.set(self.cursor)

        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        with MatrixDatabaseContextManager._lock:
            MatrixDatabaseContextManager._ref_count -= 1

            self.cursor.close()
            matrix_cursor_var.set(None)

            if (
                MatrixDatabaseContextManager._ref_count == 0
                and matrix_connection_var.get(None) is not None
            ):
                matrix_connection_var.get().close()
                matrix_connection_var.set(None)


def with_matrix_db_context(func):
    def wrapper(*args, **kwargs):
        with MatrixDatabaseContextManager() as cursor:
            # Explicitly pass the cursor as an argument to the function
            return func(*args, **kwargs, cursor=cursor)

    return wrapper


def get_matrix_connection():
    matrix_connection = sqlite3.connect("Matrix.db")

    return matrix_connection


def get_matix_cursor_for_matrix_connection(matrix_connection):
    return matrix_connection.cursor()


class Appointment:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, data):
        try:
            details = json.loads(data) if isinstance(data, str) else data
            if not isinstance(details, dict):
                raise ValueError(
                    f"Invalid data format. Expected a dictionary, got {type(details)}"
                )

            # Fetch required attributes from the database
            required_attributes = get_appointment_detail_fields()

            # Check for missing attributes
            missing_attributes = [
                attr for attr in required_attributes if attr not in details
            ]
            if missing_attributes:
                raise ValueError(
                    f"Missing required attributes: {', '.join(missing_attributes)}"
                )

            return cls(**details)
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            return None


# The get_appointment_detail_fields function should be defined as shown in the previous response


class Email:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return f"Email from {self.sender} <{self.sender_email}> received at {self.received_time}: {self.subject}"


def clean_email_content(email_content):
    # Remove URLs from the email content
    email_content = re.sub(r"http\S+", "", email_content)

    # Remove sequences of '<' possibly interspersed with whitespace and newlines
    email_content = re.sub(r"(\s*<\s*)+", " ", email_content)

    # Additional cleanup could go here if needed

    return email_content.strip()


def create_oulook_calender_appointment_for_appointment(outlook, appointment_data):
    """
    Create an appointment in Outlook from given appointment data.
    """
    # Parse the appointment data
    appointment = Appointment.from_json(appointment_data)
    if not appointment:
        print("Invalid appointment data")
        return

    namespace = outlook.GetNamespace("MAPI")
    calendar_folder = namespace.GetDefaultFolder(9)  # 9 refers to the Calendar folder

    # Create a new appointment
    new_appointment = calendar_folder.Items.Add()
    new_appointment.Subject = getattr(appointment, "subject", "No Subject")
    new_appointment.Start = getattr(appointment, "start_time", None)
    new_appointment.End = getattr(appointment, "end_time", None)
    new_appointment.Location = getattr(appointment, "location", "No Location")

    # Save the appointment
    new_appointment.Save()
    print(f"Appointment '{new_appointment.Subject}' created successfully.")


def send_email_via_outlook(outlook, subject, body, recipient):
    """Send an email using a provided Outlook instance."""
    mail = outlook.CreateItem(0)
    mail.Subject = subject
    mail.Body = body
    mail.To = recipient
    mail.Send()


def get_most_recent_unread_emails_from_outlook(outlook, folder_path=None, count=1):
    print("Connecting to Outlook...")
    namespace = outlook.GetNamespace("MAPI")

    if folder_path:
        # Navigate through the folder path
        root_folder = namespace.Folders.Item(1)  # Primary account
        target_folder = root_folder
        for folder_name in folder_path:
            target_folder = find_folder(target_folder, folder_name)
            if not target_folder:
                print(f"Folder '{folder_name}' not found in path.")
                return []
    else:
        # Default to Inbox
        print("No folder path provided. Using default Inbox...")
        target_folder = namespace.GetDefaultFolder(constants.olFolderInbox)

    print(f"Getting items from the specified folder...")
    messages = target_folder.Items
    messages.Sort("[ReceivedTime]", True)
    print("Filtering unread messages...")
    unread_messages = [
        msg for msg in messages if msg.UnRead and msg.Class == constants.olMail
    ]
    print(f"Found {len(unread_messages)} unread mail message(s).")
    emails = process_emails(unread_messages, count)
    return emails


def process_emails(messages, count):
    emails = []
    for msg in messages[:count]:
        email_obj = build_email_object(msg)
        emails.append(email_obj)
        # msg.UnRead = False  # Uncomment to mark as read
    return emails


def build_email_object(msg):
    sender_name = msg.SenderName if hasattr(msg, "SenderName") else "Unknown Sender"
    sender_email = (
        msg.SenderEmailAddress
        if hasattr(msg, "SenderEmailAddress")
        else "Unknown Email"
    )
    received_time = msg.ReceivedTime if hasattr(msg, "ReceivedTime") else "Unknown Time"
    print(
        f"Processing email from {sender_name} <{sender_email}> received at {received_time}..."
    )
    return Email(
        subject=msg.Subject,
        body=msg.Body,
        sender=sender_name,
        sender_email=sender_email,
        received_time=received_time,
    )


def get_unread_emails_from_outlook_inbox(outlook, count=1):
    return get_most_recent_unread_emails_from_outlook(outlook, count=count)


def check_email_contains_appointment(sender_email: Email) -> List[Appointment]:
    """Determine if the email is about appointments and return the details as a list."""

    client = OpenAI()

    # Fetch required fields for appointment details from the database
    required_fields = get_appointment_detail_fields()
    required_fields_str = ", ".join(
        required_fields
    )  # Convert list to a comma-separated string

    # Clean up the email content
    email_content = clean_email_content(sender_email.body)

    # Condensed prompt for the Chat API, including required fields
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Return JSON objects in response to queries about appointments. Use these fields for the JSON objects: "
            + required_fields_str
            + ".",
        },
        {
            "role": "user",
            "content": "Here is an email subject and content. Determine if it's about one or more appointments. If so, provide the details in JSON format using the specified fields.",
        },
        {"role": "user", "content": f"Subject: {sender_email.subject}"},
        {"role": "user", "content": f"Content: {email_content}"},
        {
            "role": "user",
            "content": "Carefully analyze the email for any appointments or events. Always return the details as a list in JSON format, even if there is only one appointment.",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        response_format={"type": "json_object"},
        seed=1,
        temperature=0,
        stop=["user:", "system:"],
    )

    # Access the response content
    response_text = response.choices[0].message.content.strip()

    # Convert the response text into a Python dictionary
    response_data = json.loads(response_text)

    print(response_data)

    appointments = []
    try:
        if "appointments" in response_data and isinstance(
            response_data["appointments"], list
        ):
            for appointment_data in response_data["appointments"]:
                try:
                    appointment_obj = Appointment.from_json(
                        data=json.dumps(appointment_data)
                    )
                    appointments.append(appointment_obj)
                except ValueError as e:
                    print(
                        f"Error while creating an Appointment object: {e}. Data: {appointment_data}"
                    )
        else:
            print("No appointment details found or invalid format in response.")
    except Exception as e:
        print(f"Error processing response data: {e}. Data: {response_data}")

    return appointments


@with_matrix_db_context
def add_appointment_detail_field(field_name: str, cursor):
    try:
        # Check if the field already exists
        cursor.execute(
            "SELECT COUNT(*) FROM appointment_details WHERE field_name = ?",
            (field_name,),
        )
        if cursor.fetchone()[0] == 0:
            # Insert the new field
            cursor.execute(
                "INSERT INTO appointment_details (field_name) VALUES (?)", (field_name,)
            )
            # Commit the changes
            cursor.connection.commit()
    except Exception as e:
        print("Error in add_appointment_detail_field:", e)


@with_matrix_db_context
def ensure_appointment_details_table_exists(cursor):
    # Check if the appointment_details table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='appointment_details'"
    )
    table_exists = cursor.fetchone()

    # If the table doesn't exist, create it
    if not table_exists:
        cursor.execute(
            """CREATE TABLE appointment_details (
                          id INTEGER PRIMARY KEY,
                          field_name TEXT
                          )"""
        )
        # Optionally insert default fields here


@with_matrix_db_context
def get_appointment_detail_fields(cursor):
    ensure_appointment_details_table_exists()
    # Retrieve and return all appointment detail fields
    cursor.execute("SELECT field_name FROM appointment_details")
    return [row[0] for row in cursor.fetchall()]


@with_matrix_db_context
def add_email_type(new_email_type: str, cursor):
    try:
        # Check if the email type already exists
        cursor.execute(
            "SELECT COUNT(*) FROM email_types WHERE type_name = ?", (new_email_type,)
        )
        if cursor.fetchone()[0] == 0:
            # Insert the new email type
            cursor.execute(
                "INSERT INTO email_types (type_name) VALUES (?)", (new_email_type,)
            )
            # Commit the changes
            cursor.connection.commit()
    except Exception as e:
        print("Error in add_email_type:", e)
        # Optionally, you can handle specific exceptions based on your DBMS


@with_matrix_db_context
def get_email_types_for_matrix_cursor(cursor):
    # Check if the email_types table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='email_types'"
    )
    table_exists = cursor.fetchone()

    # If the table doesn't exist, create it and insert default types
    if not table_exists:
        cursor.execute(
            """CREATE TABLE email_types (
                          id INTEGER PRIMARY KEY,
                          type_name TEXT
                          )"""
        )
        # Insert default email types
        default_types = ["Appointment", "Data Dump", "Inquiry", "Order", "Confirmation"]
        for type_name in default_types:
            cursor.execute(
                "INSERT INTO email_types (type_name) VALUES (?)", (type_name,)
            )

    # Retrieve and return all email types
    cursor.execute("SELECT type_name FROM email_types")
    return [row[0] for row in cursor.fetchall()]


def get_email_types_form_matrix():
    email_types = get_email_types_for_matrix_cursor()
    return email_types


def get_email_type_for_email(email: Email) -> Optional[str]:
    # Retrieve the current list of email types
    email_types = get_email_types_form_matrix()
    email_types_string = ", ".join(f'"{etype}"' for etype in email_types)

    print(email_types)

    # Initialize the OpenAI client
    client = OpenAI()

    # Construct the messages for the AI
    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly capable assistant specialized in email categorization. "
                "Your task is to analyze the content and subject of an email and classify it. "
                "Here are the available types: " + email_types_string + ". "
                "If the email doesn't fit any of these types, suggest a new appropriate type "
                "and present it as 'email_type' in your JSON response."
            ),
        },
        {
            "role": "user",
            "content": f"Subject: {email.subject}\nContent: {email.body}",
        },
    ]

    # Request the AI to classify the email
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        seed=1,
        temperature=0,
        response_format={"type": "json_object"},
    )

    # Extract the AI's response
    ai_response_text = response.choices[0].message.content

    # Attempt to parse the AI's response as JSON
    try:
        ai_response = json.loads(ai_response_text)
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        return None

    # Extract the email type from the AI's response
    email_type_received = ai_response.get("email_type", "").strip().lower()
    email_types_lower = [etype.lower() for etype in email_types]

    # Check if the received email type is new and add it if necessary
    if email_type_received not in email_types_lower:
        print("NEW TYPE FOUND!")
        add_email_type(email_type_received)

    # Return the email type with proper capitalization
    return email_type_received.title()


def get_read_email_from_unread_email(unread_email: Email):
    print("Checking email type...")  # Diagnostic print
    email_type = get_email_type_for_email(unread_email)

    if email_type:
        print("Email type identified:", email_type)  # Diagnostic print
    else:
        print("Email type could not be determined.")  # Diagnostic print
        return  # Early return if email type cannot be determined

    print("Checking for appointments in the email...")  # Diagnostic print
    appointments = check_email_contains_appointment(unread_email)

    if appointments:
        for appointment in appointments:
            print(appointment)
    else:
        print(
            f"No appointments in this email: {unread_email.subject}, From: {unread_email.sender}"
        )


def find_outlook_email(outlook, email_obj):
    print("Connecting to Outlook...")
    namespace = outlook.GetNamespace("MAPI")
    inbox = namespace.GetDefaultFolder(constants.olFolderInbox)
    print("Searching for the specific email...")

    for msg in inbox.Items:
        # Assuming subject, sender email, and received time are enough to uniquely identify an email
        if (
            msg.Subject == email_obj.subject
            and msg.SenderEmailAddress == email_obj.sender_email
            and msg.ReceivedTime.strftime("%Y-%m-%d %H:%M:%S")
            == email_obj.received_time.strftime("%Y-%m-%d %H:%M:%S")
        ):
            print("Matching email found.")
            return msg

    print("Email not found.")
    return None


def display_folder_tree(folder, level=0):
    """
    Recursively display the folder structure in a tree-like format.

    :param folder: The current folder to display.
    :param level: The current level in the folder hierarchy (used for indentation).
    :return: None
    """
    indent = " " * 4 * level  # 4 spaces for each level of indentation
    print(f"{indent}- {folder.Name}")

    try:
        for subfolder in folder.Folders:
            display_folder_tree(subfolder, level + 1)
    except Exception as e:
        # Ignore folders that cannot be accessed
        pass


def visualize_folder_structure(outlook):
    """
    Visualize the folder structure of an Outlook account.

    :param outlook: The outlook instance.
    :return: None
    """
    namespace = outlook.GetNamespace("MAPI")
    root_folder = namespace.Folders.Item(
        1
    )  # Usually the first item is the primary account

    print("Outlook Folder Structure:")
    for folder in root_folder.Folders:
        display_folder_tree(folder)


# Usage example
# visualize_folder_structure(outlook_instance)
def create_folder(outlook, folder_name, parent_folder):
    """
    Create a folder in Outlook within a specified parent folder.

    :param outlook: The outlook instance.
    :param folder_name: The name of the folder to be created.
    :param parent_folder: The parent folder object.
    :return: The created folder object or None if failed.
    """
    try:
        new_folder = parent_folder.Folders.Add(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
        return new_folder
    except Exception as e:
        print(f"Error creating folder '{folder_name}': {e}")
        return None


def find_folder(folder, folder_name):
    """
    Recursively search for a folder with the given name.

    :param folder: The current folder to search in.
    :param folder_name: The name of the folder to find.
    :return: The folder if found, otherwise None.
    """
    if folder.Name.lower() == folder_name.lower():
        return folder
    try:
        for subfolder in folder.Folders:
            found_folder = find_folder(subfolder, folder_name)
            if found_folder:
                return found_folder
    except Exception as e:
        # Ignore folders that cannot be accessed
        pass
    return None


def create_folders_recursive(outlook, parent_folder, structure):
    """
    Create folders and subfolders recursively based on a given structure.

    :param outlook: The outlook instance.
    :param parent_folder: The parent folder where the structure starts.
    :param structure: The folder structure defined in a dictionary.
    """
    for folder_name, sub_structure in structure.items():
        existing_folder = find_folder(parent_folder, folder_name)
        if not existing_folder:
            existing_folder = create_folder(outlook, folder_name, parent_folder)
        if existing_folder and sub_structure:  # If there are subfolders
            create_folders_recursive(outlook, existing_folder, sub_structure)


def initialize_email_folders(outlook):
    """
    Initialize the required email folders based on a JSON-defined structure.

    :param outlook: The outlook instance.
    """
    folder_structure_json = """
    {
        "User_Email_Management": {
            "Action_Required_Now": {},
            "Action_Soon": {},
            "No_Action_Required": {}
        }
    }
    """
    folder_structure = json.loads(folder_structure_json)

    root_folder = outlook.GetNamespace("MAPI").Folders.Item(1)  # Primary account
    user_email_management_folder = find_folder(root_folder, "User_Email_Management")
    if not user_email_management_folder:
        user_email_management_folder = create_folder(
            outlook, "User_Email_Management", root_folder
        )

    create_folders_recursive(
        outlook, user_email_management_folder, folder_structure["User_Email_Management"]
    )


def set_email_folder_for_outlook_email(outlook_email, folder_path, outlook):
    """
    Move an email to the specified folder based on the provided path.

    :param email: The email object to be moved.
    :param folder_path: A list representing the path to the destination folder.
    :param outlook: The outlook instance.
    """
    namespace = outlook.GetNamespace("MAPI")
    root_folder = namespace.Folders.Item(1)  # Primary account

    # Navigate through the folder path
    target_folder = root_folder
    for folder_name in folder_path:
        target_folder = find_folder(target_folder, folder_name)
        if not target_folder:
            print(f"Folder '{folder_name}' not found in path.")
            return

    # Move the email
    try:
        outlook_email.Move(target_folder)
        print(f"Email moved to '{' > '.join(folder_path)}'.")
    except Exception as e:
        print(f"Error moving email: {e}")


def determine_email_priority(sender_email: Email) -> str:
    """Determine the priority of the email and categorize it into the appropriate folder based on detailed criteria."""

    client = OpenAI()

    # Clean up the email content
    email_content = clean_email_content(sender_email.body)

    # Get the current date and time
    current_time_and_date = get_current_time_and_date()

    # Detailed instructions for the AI to categorize the email
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Analyze the email and categorize it as 'Action_Required_Now', 'Action_Soon', or 'No_Action_Required'. Use specific criteria for each category. Consider the context of the email, including the sender's role and previous communications. Confirm your decision before finalizing. Return the category in a simplified JSON format like {'category': 'Action_Required_Now'}. Handle uncertain cases with a specific procedure and collect feedback for continuous improvement. Consider the current date and time: {current_time_and_date}."
        },
        {
            "role": "user",
            "content": "Here is an email subject and content. Determine its priority and categorize it accordingly."
        },
        {"role": "user", "content": "Subject: {sender_email.subject}"},
        {"role": "user", "content": "Content: {email_content}"}
    ]


    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        seed=1,
        temperature=0,
        response_format={"type": "json_object"},
        stop=["user:", "system:"],
    )

    # Access the response content
    response_text = response.choices[0].message.content.strip()

    # Convert the response text into a Python dictionary
    response_data = json.loads(response_text)

    # Determine the priority category
    priority_category = response_data.get("category", "No_Action_Required")

    return priority_category


def get_current_time_and_date():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    initialize_email_folders(outlook)

    # visualize_folder_structure(outlook)

    # outlook = win32.Dispatch("Outlook.Application")
    unread_emails = get_unread_emails_from_outlook_inbox(
        outlook, count=40
    )  # Assuming this function returns a list of Email objects
    for unread_email in unread_emails:
        email_priority = determine_email_priority(unread_email)
        outlook_email = find_outlook_email(outlook, unread_email)
        folder_path = ["User_Email_Management", email_priority]
        set_email_folder_for_outlook_email(outlook_email, folder_path, outlook)

    # read_email = get_read_email_from_unread_email(unread_email)
    # Check if the email is about an appointment and get the details

    # Test sending an email
    # subject = "Test Email from AI Hub"
    # body = "This is a test email sent from the AI Hub using local Outlook instance."
    # recipient = "notmymail@outlook.de"
    # send_email_via_outlook(subject, body, recipient)
