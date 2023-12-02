import os
import openai
import dotenv


dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


#print("Books in pdf folder:")
#for file in os.listdir("pdfs"):
#    print(file)
#
#print("-----------------------------------")

print("Folders in current directory")
for file in os.listdir():
    if os.path.isdir(file) and file != "pdfs":
        print(file)



folder_name = input("Enter the name of the folder you want to extract text from: ")



#get the list of all files in the directory page_chunks
files = os.listdir(f"{folder_name}/page_chunks")




start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

if not os.path.exists(f"{folder_name}/flash_chunks"):
    os.makedirs(f"{folder_name}/flash_chunks")

if not os.path.exists(f"{folder_name}/file_exceptions"):
    os.makedirs(f"{folder_name}/file_exceptions")  


for index, file in enumerate(files[2:]):
    
    try:
        file_content = open(f"{folder_name}/page_chunks/{file}", "r", encoding="utf-8").read()


        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages= [{"role": "system", "content": "You produce flashcards from a two-page section from a book. You produce highly detailed flash cards with a term name and a definition in the format: Term: <Card Name> \n Definition: <Card Definition> \n ... Term: <Card Name> \n Defnition: <Card Definition>"}, {"role": "user", "content": "please produce some flashcards from the provided content: \n" + file_content}],
        )

        with open(f"{folder_name}/flash_chunks/flash_chunks_{index}.txt", "w", encoding="utf-8") as f:
            f.write(response["choices"][0]["message"]["content"])

    except Exception as e:

        # write file exepction to a file in file_exceptions folder
        with open(f"{folder_name}/file_exceptions/{file}.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
        



    




