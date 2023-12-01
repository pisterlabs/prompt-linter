from openai import OpenAI
import datetime
import os

client = OpenAI()
thread = client.beta.threads.create()

def get_assistants():

    assistant_list = client.beta.assistants.list(
        order="desc",
        limit="20"
    )
        
    i = 1
    while assistant_list.has_more:
        print(i)
        assistants = client.beta.assistants.list(
            order="desc",
            limit=str(20 * i)
        )
        assistant_list.append(assistants.data)
        i += 1

    return assistant_list

def get_file(file_id):
    file = client.files.retrieve(file_id)

    return file

def select_file_to_update(json_data):
    # Function to convert epoch time to human-readable format
    def epoch_to_human_readable(epoch_time):
        return datetime.datetime.fromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M:%S')

    # Display the assistant names and their files with indices
    file_indices = {}
    current_index = 1

    for asst_id, asst_data in json_data.items():
        print(f"Assistant: {asst_data['name']}")
        for file_info in asst_data.get('files', []):
            file_name = file_info['filename']
            file_id = file_info['file_id']
            last_update = epoch_to_human_readable(file_info['last_update'])
            print(f"  {current_index}: {file_name} (Last Update: {last_update})")
            file_indices[current_index] = (asst_id, file_id, file_name)
            current_index += 1

    # Ask the user to select a file
    try:
        selected_index = int(input("Enter the number of the file you want to update: "))
        if selected_index in file_indices:
            return file_indices[selected_index]
        else:
            print("Invalid selection. Please try again.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

def handle_file(assistant_id, file_id, filename):
    # Delete the old file
    delete_file = client.files.delete(
        file_id
    )
    # If the file was successfully deleted
    if delete_file.deleted:
        try:
            # Upload the new file
            file_content = open(filename, "rb")
            
            file = client.files.create(
                file=file_content,
                purpose="assistants"
            )

            # Get list of current files so we can ensure we dont overwrite other files
            current_assistant = client.beta.assistants.retrieve(assistant_id)
            
            file_ids = [file.id]

            # Get the list of current file ids so we dont clobber them
            existing_file_ids = current_assistant.file_ids
            for identifier in existing_file_ids:
                if identifier == file_id:
                    continue
                else:
                    file_ids.append(identifier)
            # Associate the new file to the current assistant
            updated_assistant = client.beta.assistants.update(
                assistant_id,
                file_ids=file_ids
            )

            return updated_assistant
        except:
            return None

def main():
    # Define our main dict
    assistant_to_file_mappings = {}

    # Get our list of assistants
    assistants = get_assistants()

    # Iterate through them
    for assistant in assistants:
        assistant_object = {'name': assistant.name}
        files = []

        # For each file id, get its filename
        for file_id in assistant.file_ids:
            file_details = {}
            file = get_file(file_id)
            file_details['file_id'] = file.id
            file_details['filename'] = file.filename
            file_details['last_update'] = file.created_at
            files.append(file_details)
        # Add the files dict to our assistant_object dict
        assistant_object['files'] = files

        # Add this object to our main dict
        assistant_to_file_mappings[assistant.id] = assistant_object
        
    # Iterate through our mappings
    while True:
        selected_file = select_file_to_update(assistant_to_file_mappings)  # Assuming json_data is defined
        if selected_file:
            assistant_id, file_id, filename = selected_file
            confirm = input(f"Confirm file selection: {filename} (Y/n): ").lower() or 'y'
            if confirm not in ['y', 'yes']:
                continue

            # Listing files in the current directory
            print("Files in current directory:")
            files_in_dir = [f for f in os.listdir('.') if os.path.isfile(f)]
            for i, file in enumerate(files_in_dir):
                print(f"{i + 1}: {file}")

            # Selecting a file to update
            try:
                file_index = int(input("Select a file to update by entering its number: ")) - 1
                if 0 <= file_index < len(files_in_dir):
                    # Call the function to update the file using OpenAI API
                    updated_assistant = handle_file(assistant_id, file_id, files_in_dir[file_index])
                    if updated_assistant:
                        current_files = updated_assistant.file_ids
                        assistant_object = {'name': updated_assistant.name}
                        files = []
                        for file_id in current_files:
                            file_details = {}
                            file = get_file(file_id)
                            file_details['file_id'] = file.id
                            file_details['filename'] = file.filename
                            file_details['last_update'] = file.created_at
                            files.append(file_details)
                        assistant_object['files'] = files
                        assistant_to_file_mappings[updated_assistant.id] = assistant_object
                        print(f"File updated successfully.")
                    else:
                        print('[-] File failed to update.')
                else:
                    print("Invalid file selection.")
            except ValueError:
                print("Please enter a valid number.")

            # Check if the user wants to update more files
            update_more = input("Do you want to update more files? (Y/n): ").lower() or 'y'
            if update_more not in ['y', 'yes']:
                break

if __name__ == '__main__':
    main()
