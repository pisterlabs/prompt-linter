import os
import openai
import sv_ttk
import tkinter
from datetime import datetime
from tkinter import *
from tkinter import ttk
from tkinter import font

openai.api_key = ''

output_file_path = "gui_output.txt"

if not os.path.exists(output_file_path):
    with open(output_file_path, "w") as file:
        # Perform any initial setup or write any default content to the file if needed
        file.write("Initial content")
    print("File created successfully!")
else:
    pass

root = tkinter.Tk()
root.title("GUI GPT")

root.state('zoomed')

predefined_content = {
    "Create Perfect Prompt": "",

    "Correct Grammar": "",

    "Debug Python Code": "",

    "Write me Python code": "",

    "Legal Tutor": "",

    "Legal Assistant/Consultant/Expert/Advisor": "",

    "Programming/Computer Science Tutor": "",

    "Paraphrase a text": "",


}

def get_system_content(option):
    return predefined_content.get(option, "No matching content found.")

def clean_message(message):
    return message.replace("\n", " ").replace("\r", "")

def update_print_box(output):
    print_box.config(state='normal')
    print_box.delete('1.0', 'end')
    print_box.insert('1.0', output)
    print_box.config(state='disabled')

def send_button_api():
    selected_option = radiobutton_var.get()

    system_content = get_system_content(selected_option)

    user_content = text.get(1.0, 'end-1c')

    input_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    completion = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=input_messages
)   
    current_time = datetime.now()

    timestamp = current_time.strftime("%d/%b/%Y %H:%M:%S")
    
    model_output = completion.choices[0].message['content']

    # print(model_output)

    update_print_box(model_output)

    cleaned_input_messages = [clean_message(msg['content']) for msg in input_messages] #This is user input
    cleaned_output_message = clean_message(model_output)


    with open(output_file_path, "a") as output_file:
        output_file.write("Timestamp: " + timestamp + "\n")
        output_file.write("Input Messages:\n")

        for msg in cleaned_input_messages:
            output_file.write(f"{msg}\n")
        output_file.write("\nOutput:\n")
        output_file.write(cleaned_output_message)
        output_file.write("\n\n")
        output_file.write("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        output_file.write("\n\n")
    print("Output written to", output_file_path)

button_style = ttk.Style()
button_style.configure("Rounded.TButton", padding=(10, 5),
                       borderwidth=0, relief="solid", 
                       bordercolor="black", background="lightblue", 
                       foreground="black", borderradius=10)

main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=True)

radiobutton_frame = ttk.Frame(main_frame)
radiobutton_frame.pack(side='top', fill=BOTH, expand=True, padx=10)

ui_frame = Frame(main_frame)
ui_frame.pack(side='top', fill=BOTH, expand=True)

textbox_frame = Frame(ui_frame)
textbox_frame.pack(side='left', fill=BOTH, expand=True)

button_frame = Frame(ui_frame)
button_frame.pack(side='left')

printbox_frame = Frame(ui_frame)
printbox_frame.pack(side='right', fill=BOTH, expand=True)

checkbox_options = ["Create a Perfect Prompt", "Prepare a RTI Draft", "File first appeal for RTI",
                    "Correct Grammar", "Debug the Python Code", "Write Python code for me", 
                    "Legal Tutor", "Legal Assistant/Consultant/Expert/Advisor", 
                    "Programming/Computer Science Tutor", "Paraphrase a text"
                    ]
                    
radiobutton_var = tkinter.StringVar(value="Create Perfect Prompt")
radiobutton_var = tkinter.StringVar(value="Prepare a RTI Draft")
radiobutton_var = tkinter.StringVar(value="File first appeal for RTI")
radiobutton_var = tkinter.StringVar(value="Correct Grammar")
radiobutton_var = tkinter.StringVar(value="Debug Python Code")
radiobutton_var = tkinter.StringVar(value="Write me Python code")
radiobutton_var = tkinter.StringVar(value="Legal Tutor")
radiobutton_var = tkinter.StringVar(value="Legal Assistant/Consultant/Expert/Advisor")
radiobutton_var = tkinter.StringVar(value="Programming/Computer Science Tutor")
radiobutton_var = tkinter.StringVar(value="Paraphrase a text")

for option in predefined_content.keys():
    rb = ttk.Radiobutton(radiobutton_frame, text=option, variable=radiobutton_var, value=option)
    rb.pack(side='left', padx=10)

user_label = ttk.Label(textbox_frame, text="Enter your text here:")
user_label.pack(pady=(50, 0))  # Set padding for the label

text = Text(textbox_frame, width=150, height=75)
text.pack(padx=10)

quit_button = ttk.Button(button_frame, text="Quit",command=root.destroy, style="Rounded.TButton")
quit_button.pack(side='right', padx=10)

send_button = ttk.Button(button_frame, text="Send",command=send_button_api, style="Rounded.TButton")
send_button.pack(side='right', padx=10)

response_label = ttk.Label(printbox_frame, text="Response recieved:")
response_label.pack(pady=(50, 0))

print_box = Text(printbox_frame, width=150, height=75, state='disabled')  # Make it read-only
print_box.pack(padx=10)

sv_ttk.set_theme("dark")
root.mainloop()
