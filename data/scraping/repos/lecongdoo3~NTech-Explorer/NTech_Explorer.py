####
# Cong Do Le - S366422
####
# Import necessary libraries
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk, Text, Toplevel
from transformers import BartForConditionalGeneration, BartTokenizer
import webbrowser
from PIL import Image, ImageTk
import openai

# Load the model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Set the OpenAI API key
openai.api_key = "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"

def toggle_password():
    """Toggle between masked and unmasked password."""
    if check_var.get():
        entry_2.config(show="")
    else:
        entry_2.config(show="*")

def log_in():
    """Handle user login."""
    username = entry_1.get()
    password = entry_2.get()

    # Check if credentials match any account in the accounts.txt file
    with open("accounts.txt", "r") as f:
        accounts = f.readlines()
        for account in accounts:
            id, pwd = account.strip().split(',')
            if username == id and password == pwd:
                # Close the login window
                root.destroy()
                # Launch the main application
                run_educational_system()
                return

    # Display error message if credentials don't match
    messagebox.showerror("Error", "ID or Password is incorrect. Please try again!")
        
def run_educational_system():
    """Launch the main application (ChatBot)."""
    
    class ChatBot:
        """Defines the ChatBot GUI and its functionalities."""
        
        def __init__(self, root):
            """Initialize the ChatBot UI."""           
            # Set up the main frame at the bottom of the root window
            self.frame = tk.Frame(root)
            self.frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Entry widget for user to type in their messages
            self.user_entry = tk.Entry(self.frame)
            self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Bind the Enter key to the send_message function
            self.user_entry.bind("<Return>", self.send_message_by_event)
            
            # Text widget to display the chat history; height is doubled for better visibility
            self.chat_history = tk.Text(root, wrap=tk.WORD, width=70, height=12)
            self.chat_history.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
            
            # Prevent the user from editing the text inside the chat history
            self.chat_history.bind("<Key>", self.prevent_bot_edit)

            # Configure a tag for bot messages with a yellow background
            self.chat_history.tag_configure("bot", background="yellow")
            
            # Button to send the user's message
            self.send_button = tk.Button(self.frame, text="Send", command=self.send_message)
            self.send_button.pack(side=tk.LEFT)
            
            # Button to start a new conversation
            self.new_conversation_button = tk.Button(self.frame, text="New Conversation", command=self.new_conversation)
            self.new_conversation_button.pack(side=tk.LEFT)

            # Initial message from the bot when the chat starts
            self.chat_history.insert(tk.END, "Bot: How can I help you?\n", "bot")

        def get_bot_response(self, prompt):
            """Get a response from the bot based on the user's prompt."""
            # Use the openai API to get a response for the given prompt
            self.response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the assistant's response from the returned data
            assistant_response = self.response['choices'][0]['message']['content']
            return assistant_response
        
        def prevent_bot_edit(self, event):
            """Prevent the user from editing the bot's messages."""
            # Check if the current selection in the chat history has the "bot" tag
            if "bot" in self.chat_history.tag_names(tk.CURRENT):
                return "break"
        
        def send_message_by_event(self, event):
            """Send a message when the Enter key is pressed."""
            # Since this function is called by an event binding, the event parameter will be passed.
            # However, the rest of the function will be the same as send_message.
            self.send_message()
        
        def send_message(self):
            """Send user's message and display bot's reply."""
            user_input = self.user_entry.get().strip()  # Read from the Entry widget
            self.user_entry.delete(0, tk.END)  # Clear the Entry widget
            user_input = ' '.join(user_input.split()[:50])
            if user_input:
                self.chat_history.insert(tk.END, f"\nYou: {user_input}\n", "user")
                
                # Call the function to get the bot's reply
                bot_reply = self.get_bot_response(user_input)
                self.chat_history.insert(tk.END, f"Bot: {bot_reply}\n", "bot")

        def new_conversation(self):
            """Start a new conversation with the bot."""
            self.chat_history.delete(1.0, tk.END)
            self.chat_history.insert(tk.END, "Bot: How can I help you?\n", "bot")

        def chat_response(self, user_input):
            """Get response for a given user input."""
            try:
                # Call the function to get response from chatbot
                assistant_response = self.get_bot_response(user_input)
                return assistant_response
            except Exception as e:
                # Handle any exceptions that arise
                return str(e)  # Return exception as string for debugging

    class SubjectTab:
        """A class to represent the subject tab in the GUI."""
        def __init__(self, parent, subject_info):
            """Initialize the SubjectTab with the provided subject info."""
            self.frame = ttk.Frame(parent)
            self.title = subject_info["title"]
            self.image_path = subject_info["image"]
            self.video_url = subject_info["video_url"]
            self.document_url = subject_info["document_url"]
            self.quiz_url = subject_info["quiz_url"]
            self.materials = []
            self._init_tab()
            
        def _init_tab(self):
            """Set up the GUI elements for the tab."""
            # Add image
            subject_image = Image.open(self.image_path)
            subject_image = subject_image.resize((390, 140), Image.ANTIALIAS)
            subject_photo = ImageTk.PhotoImage(subject_image)
            subject_label = tk.Label(self.frame, image=subject_photo)
            subject_label.image = subject_photo
            subject_label.pack(anchor='w')

            # Add level dropdown
            self.level_var = tk.StringVar()
            self.level_dropdown = ttk.Combobox(self.frame, textvariable=self.level_var, state="readonly", width=61)
            self.level_dropdown['values'] = ('Pre-school', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5', 'Year 6',
                                            'Year 7', 'Year 8', 'Year 9', 'Year 10', 'Year 11', 'Year 12')
            self.level_dropdown.pack()
            
            # Set default value for the dropdown
            self.level_var.set('Pre-school')
            # Call show_materials for the default value
            self.show_materials(None)
            
            self.level_dropdown.bind('<<ComboboxSelected>>', self.show_materials)

        def clear_materials(self):
            """Clear displayed materials from the tab."""
            for material in self.materials:
                material.destroy()
            self.materials = []
        
        def show_materials(self, event=None):
            """Display materials for the selected level."""
            # Clear any previously displayed materials
            self.clear_materials()

            # Get the selected level from the associated variable
            level = self.level_var.get()         
            
            # Create and pack a label indicating the materials for the selected level
            level_label = ttk.Label(self.frame, text=f"Materials for {self.title} at {level}")
            level_label.pack()
            self.materials.append(level_label) # Add the label to the materials list to keep track of it

            # Create, pack, and add a button to open tutorial videos
            btn_video = tk.Button(self.frame, text="Tutorial Videos", command=self.open_video)
            btn_video.pack()
            self.materials.append(btn_video)

            # Create, pack, and add a button to open learning resources
            btn_document = tk.Button(self.frame, text="Learning Resources", command=self.open_document)
            btn_document.pack()
            self.materials.append(btn_document)

            # Create, pack, and add a button to open quiz games
            btn_quiz = tk.Button(self.frame, text="Quiz Games", command=self.open_quiz)
            btn_quiz.pack()
            self.materials.append(btn_quiz)

        def open_video(self):
            """Open the video link in a browser."""
            level = self.level_var.get()
            video_url = self.video_url.get(level)  # Get the appropriate video URL based on the selected level
            if video_url:
                webbrowser.open(video_url)
            else:
                messagebox.showinfo("Info", f"No video available for {self.title} at {level}.")

        def open_document(self):
            """Open the document link in a browser."""
            level = self.level_var.get()
            document_url = self.document_url.get(level)  # Get the appropriate document URL based on the selected level
            if document_url:
                webbrowser.open(document_url)
            else:
                messagebox.showinfo("Info", f"No document available for {self.title} at {level}.")
                
        def open_quiz(self):
            """Open the quiz link in a browser."""
            level = self.level_var.get()
            quiz_url = self.quiz_url.get(level)  # Get the appropriate quiz games URL based on the selected level
            if quiz_url:
                webbrowser.open(quiz_url)
            else:
                messagebox.showinfo("Info", f"No quiz games available for {self.title} at {level}.")                
        
    # GUI Design
    root = tk.Tk()
    root.title("Online Educational System - NTech Explorer")
    root.geometry("1000x790")

    # Main Frame which will contain both the tabControl and the image
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # Split main_frame into top and bottom
    top_frame = tk.Frame(main_frame)
    top_frame.grid(row=0, column=0, sticky="nsew")

    bottom_frame = tk.Frame(main_frame)
    bottom_frame.grid(row=1, column=0, sticky="nsew")

    # Frame for the image inside the top_frame
    img_frame = tk.Frame(top_frame)
    img_frame.pack(side='right', fill="y")

    # Load and display the image inside the img_frame
    image = Image.open("back_ground_img.jpg")
    image = image.resize((600, 600), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    # Dimensions of the canvas (which matches the image dimensions)
    canvas = tk.Canvas(img_frame, width=600, height=600)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # List of news titles and their respective URLs
    news_list = [
        {"title": "Nganambala & Gunbalanya Students Tackle Issues with Space Tech", "url": "https://www.abc.net.au/news/2023-06-16/nt-remote-students-space-technology-payphone/102485136"},
        {"title": "NT Government to Invest in Digital Connectivity for Territory Schools", "url": "https://www.miragenews.com/nt-government-to-invest-in-digital-connectivity-1005930/"},
        {"title": "Tech over time - celebrating 40 years", "url": "https://www.ntit.net.au/news-byte/tech-over-time-celebrating-40-years"},
        {"title": "NT's Power and Water maps out huge core IT upgrade", "url": "https://www.itnews.com.au/news/nts-power-and-water-maps-out-huge-core-it-upgrade-594244"},
        {"title": "New technology aims to shore up Australia's food supply", "url": "https://www.abc.net.au/news/2023-06-23/new-technology-aims-to-shore-up-australias-food-supply/102516148"},
        {"title": "The state of STEM gender equity in 2022", "url": "https://innovation.nt.gov.au/news/2022/the-state-of-stem-gender-equity-in-2022"},
        {"title": "HyperOne to supercharge Territory economy", "url": "https://innovation.nt.gov.au/news/2022/hyperone-to-supercharge-territory-economy"},
    ]

    # List of article titles and their respective URLs
    articles = [
        {"title": "Artificial Intelligence for Science (November 2022)", "url": "https://www.csiro.au/-/media/D61/AI4Science-report/Plain-text-AI-for-Science-report-2022.txt"},
        {"title": "Strengthening Australian Aboriginal Participation in University STEM Programs", "url": "https://www.tandfonline.com/doi/full/10.1080/07256868.2018.1552574?scroll=top&needAccess=true&role=tab"},
        {"title": "Tools to Boost Public Trust in the Howard East Aquifer Plan, Northern Territory", "url": "https://www.sciencedirect.com/science/article/pii/S0022169412001059"},
        {"title": "Using computer-based instruction to improve Indigenous early literacy", "url": "https://ajet.org.au/index.php/AJET/article/view/947/223"},
        {"title": "Managing Indigenous Digital Data: an exploration of the Our Story database, NT", "url": "https://opus.lib.uts.edu.au/bitstream/10453/19485/1/2009%20-%20Gibson.pdf"},
        {"title": "Indigenous Knowledge Sharing: Digital Technology for NT's Cultural Interpretation", "url": "https://www.tandfonline.com/doi/epdf/10.1080/21568316.2019.1704855?needAccess=true&role=button"},
        {"title": "ICT and New Indigenous Mobilities? Insights From Remote NT Communities", "url": "https://www.researchgate.net/publication/267639886_Information_communication_technologies_and_new_Indigenous_mobilities_Insights_from_remote_Northern_Territory_Communities"},
    ]

    # Function to open a web link
    def open_link(url):
        webbrowser.open(url)

    # Function to add underline to text on hover
    def on_hover(event, text_id):
        canvas.itemconfig(text_id, underline=True)

    # Function to add underline to text on hover
    def on_hover(event, text_id):
        canvas.itemconfig(text_id, font=("Arial", 12, "underline"))

    # Function to remove underline from text when not hovering
    def on_leave(event, text_id):
        canvas.itemconfig(text_id, font=("Arial", 12))

    # Dimensions of the canvas (which matches the image dimensions)
    canvas_width = 600
    canvas_height = 600

    # Create the main title for the list of articles, centered on the canvas
    news_title_text = "Recent news and reports about science and technology in the NT"
    news_title_pos_x = canvas_width / 2
    news_title_pos_y = 20  # Adjust this to move the title up or down
    canvas.create_text(news_title_pos_x, news_title_pos_y, anchor=tk.N, text=news_title_text, fill="yellow", font=("Arial", 13, "bold"))

    # Place each news title on the canvas
    news_y_pos = 60  # Initial y-coordinate for the first news title
    news_x_pos = 30  # Initial x-coordinate for the first news title

    # Loop through all the news in the list
    for news in news_list:
        text_id = canvas.create_text(news_x_pos, news_y_pos, anchor=tk.NW, text=news["title"], fill="black", font=("Arial", 11))
        
        # Bind the click event on the text to open the link
        canvas.tag_bind(text_id, "<Button-1>", lambda e, link=news["url"]: open_link(link))
        
        # Bind hover events to the text
        canvas.tag_bind(text_id, "<Enter>", lambda e, tid=text_id: on_hover(e, tid))
        canvas.tag_bind(text_id, "<Leave>", lambda e, tid=text_id: on_leave(e, tid))
        
        news_y_pos += 30  # Increment the y-coordinate for the next title
        
    # Create the main title for the list of articles, centered on the canvas
    article_title_text = "Recent academic articles about science and technology in the NT"
    article_title_pos_x = canvas_width / 2
    article_title_pos_y = 290  # Adjust this to move the title up or down
    canvas.create_text(article_title_pos_x, article_title_pos_y, anchor=tk.N, text=article_title_text, fill="blue", font=("Arial", 13, "bold"))

    # Place each article title on the canvas
    article_y_pos = 320  # Initial y-coordinate for the first article title
    article_x_pos = 30  # Initial x-coordinate for the first article title

    # Loop through all the articles
    for article in articles:
        text_id = canvas.create_text(article_x_pos, article_y_pos, anchor=tk.NW, text=article["title"], fill="white", font=("Arial", 11))
        
        # Bind the click event on the text to open the link
        canvas.tag_bind(text_id, "<Button-1>", lambda e, link=article["url"]: open_link(link))
        
        # Bind hover events to the text
        canvas.tag_bind(text_id, "<Enter>", lambda e, tid=text_id: on_hover(e, tid))
        canvas.tag_bind(text_id, "<Leave>", lambda e, tid=text_id: on_leave(e, tid))
        
        article_y_pos += 30  # Increment the y-coordinate for the next title

    # Create text for further readings
    further_reading_text = {"title": "Further Readings", "url": "https://www.csiro.au/en/research/technology-space"}
    
    further_reading_pos_x = canvas_width / 2
    further_reading_pos_y = 550  # Adjust this to move the title up or down
    further_reading = canvas.create_text(further_reading_pos_x, further_reading_pos_y, anchor=tk.N, text=further_reading_text["title"], fill="brown", font=("Arial", 11, "bold"))
    
    # Bind the click event on the text to open the link
    canvas.tag_bind(further_reading, "<Button-1>", lambda e, link=news["url"]: open_link(link))
    
    # Bind hover events to the text
    canvas.tag_bind(further_reading, "<Enter>", lambda e, tid=further_reading: on_hover(e, tid))
    canvas.tag_bind(further_reading, "<Leave>", lambda e, tid=further_reading: on_leave(e, tid))
    
    # Add the tabControl to the top_frame
    tabControl = ttk.Notebook(top_frame)
    tabControl.pack(fill="both", expand=True)

    def summarize_text():
        # Create a new Toplevel window
        summary_window = Toplevel()
        summary_window.title("Text Summarizer")
        summary_window.geometry("600x400")  # You can adjust the size
        
        # Input Text widget for pasting long text
        input_text = Text(summary_window, wrap=tk.WORD, height=10)  # Adjust the height as needed
        input_text.pack(pady=10, padx=10, fill=tk.X)

        # Button to trigger summarization
        summarize_button = ttk.Button(summary_window, text="Summarize", command=lambda: display_summary(input_text, output_text))
        summarize_button.pack(pady=10)

        # Output Text widget for displaying summarized text
        output_text = Text(summary_window, wrap=tk.WORD, height=10, bg="#EEE")  # Adjust the height and other properties as needed
        output_text.pack(pady=10, padx=10, fill=tk.X)
        
    def summarise(input_text):
        # Summarize using the BART model
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=250, min_length=150, length_penalty=5., num_beams=2)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def display_summary(input_widget, output_widget):
        original_text = input_widget.get(1.0, tk.END)
        summarized = summarise(original_text)
        output_widget.delete(1.0, tk.END)
        output_widget.insert(tk.END, summarized)
        
    # Create the button and position it
    button_width = 100  # Width of the button, you can adjust as per your design
    button_height = 25  # Height of the button, you can adjust as per your design
    button_x_pos = further_reading_pos_x + button_width / 2 + 195  # Positioned next to 'Further Readings' with a gap of 10 units
    button_y_pos = further_reading_pos_y + 25

    summarise_button = tk.Button(canvas, text="Summarise Text", command=summarize_text)
    canvas.create_window(button_x_pos, button_y_pos, anchor=tk.N, window=summarise_button, width=button_width, height=button_height)

    
    subjects = [
        # Math
        {"title": "Math", "image": "math.jpg",
        "video_url": {'Pre-school': "https://www.youtube.com/playlist?list=PL657ZRLHPzqSPxHmFLTs6lE51d8clcj5V",
                      'Year 1': "https://www.youtube.com/playlist?list=PLIHsR1xaQUW29wtx6lD2a5egPlDZDhSDQ",
                      'Year 2': "https://www.youtube.com/playlist?list=PLVG8tGc7EksbHRd8d7G2eIIguX5PbO2fF",
                      'Year 3': "https://www.youtube.com/playlist?list=PLVG8tGc7Eksb4rS8H6brqzlJ2GQmpJWqZ",
                      'Year 4': "https://www.youtube.com/playlist?list=PLVG8tGc7EksbibMrpNflbyX8XYjGEkBW4",
                      'Year 5': "https://www.youtube.com/playlist?list=PL1E355n62j7VgkrYnHvukpCgjvSw8vU_a",
                      'Year 6': "https://www.youtube.com/playlist?list=PLdTPQ62ogX0aFBxsWB8uB2OjpoRGH94-J",
                      'Year 7': "https://www.youtube.com/playlist?list=PL-eep-5iLk017hF9sOFzOwg9ZMnHBYoOG",
                      'Year 8': "https://www.youtube.com/playlist?list=PLdTPQ62ogX0Y9YlYRpV7FShhgWDme9nz3",
                      'Year 9': "https://www.youtube.com/playlist?list=PLdTPQ62ogX0Ze30MbL-LJ761m33KAlZLT",
                      'Year 10': "https://www.youtube.com/playlist?list=PLdTPQ62ogX0Yk0co0u54Rzcc2bsOjEVyZ",
                      'Year 11': "https://www.youtube.com/playlist?list=PLdTPQ62ogX0Yn_e8XL_63ZFoCFI01bat1",
                      'Year 12': "https://www.youtube.com/playlist?list=PLdTPQ62ogX0aQRvPBzadGQlnNzc-bZ9ne"},
        "document_url": {'Pre-school': "https://au.ixl.com/maths/preschool",
                         'Year 1': "https://au.ixl.com/maths/year-1",
                         'Year 2': "https://au.ixl.com/maths/year-2",
                         'Year 3': "https://au.ixl.com/maths/year-3",
                         'Year 4': "https://au.ixl.com/maths/year-4",
                         'Year 5': "https://au.ixl.com/maths/year-5",
                         'Year 6': "https://au.ixl.com/maths/year-6",
                         'Year 7': "https://au.ixl.com/maths/year-7",
                         'Year 8': "https://au.ixl.com/maths/year-8",
                         'Year 9': "https://au.ixl.com/maths/year-9",
                         'Year 10': "https://au.ixl.com/maths/year-10",
                         'Year 11': "https://au.ixl.com/maths/year-11",
                         'Year 12': "https://au.ixl.com/maths/year-12"},
        "quiz_url": {'Pre-school': "https://www.education.com/games/preschool/math/",
                         'Year 1': "https://www.education.com/games/first-grade/math/",
                         'Year 2': "https://www.education.com/games/second-grade/math/",
                         'Year 3': "https://www.education.com/games/third-grade/math/",
                         'Year 4': "https://www.education.com/games/fourth-grade/math/",
                         'Year 5': "https://www.education.com/games/fifth-grade/math/",
                         'Year 6': "https://www.education.com/games/sixth-grade/math/",
                         'Year 7': "https://www.education.com/games/seventh-grade/math/",
                         'Year 8': "https://www.education.com/games/eighth-grade/math/",
                         'Year 9': "https://ezymathtutoring.com.au/resources/maths-test/year-9/",
                         'Year 10': "https://ezymathtutoring.com.au/resources/maths-test/year-10/",
                         'Year 11': "https://mathsmethods.com.au/courses/free-maths-methods-exam-questions/",
                         'Year 12': "https://mathsmethods.com.au/courses/free-maths-methods-exam-questions/"}},
        
        # Science
        {"title": "Science", "image": "science.jpg",
        "video_url": {'Pre-school': "https://www.youtube.com/watch?v=hvHAtMzMm5g",
                      'Year 1': "https://www.youtube.com/playlist?list=PLoPH9JUqy7ER1YyneEyxCeKy19yss8gBq",
                      'Year 2': "https://www.youtube.com/playlist?list=PLVG8tGc7EksZbbJ7TpRzsylU5uQXD7oqT",
                      'Year 3': "https://www.youtube.com/playlist?list=PLVG8tGc7EksaYgVo9fhqjm_nR559cF81T",
                      'Year 4': "https://www.youtube.com/playlist?list=PLVG8tGc7EksYWvm1ygS8IHPTFXuEl1JGt",
                      'Year 5': "https://www.youtube.com/playlist?list=PLHUCnNctgKrSUsf-GkmiuW-n5LBHzS-E8",
                      'Year 6': "https://www.youtube.com/playlist?list=PLNCvuwUbSb7IjPxQt_Xn60_OJ_gUPA9WP",
                      'Year 7': "https://www.youtube.com/playlist?list=PL6-lV1qXhb0jtn5Pprr-nYTcVGW71ENf4",
                      'Year 8': "https://www.youtube.com/playlist?list=PLJ8TC1CA7IZNQ3bUz3PefQpAiqR6sd0gl",
                      'Year 9': "https://www.youtube.com/playlist?list=PLPtMjavUFUUSHnhbpMQS4br9Zv7sdBejG",
                      'Year 10': "",
                      'Year 11': "",
                      'Year 12': ""},
        "document_url": {'Pre-school': "",
                         'Year 1': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-1/",
                         'Year 2': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-2/",
                         'Year 3': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-3/",
                         'Year 4': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-4/",
                         'Year 5': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-5/",
                         'Year 6': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-6/",
                         'Year 7': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-7/",
                         'Year 8': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-8/",
                         'Year 9': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-9/",
                         'Year 10': "https://www.australiancurriculumlessons.com.au/category/science-lessons/science-lessons-year-10/",
                         'Year 11': "https://senior-secondary.scsa.wa.edu.au/syllabus-and-support-materials/science",
                         'Year 12': "https://senior-secondary.scsa.wa.edu.au/syllabus-and-support-materials/science"},
        "quiz_url": {'Pre-school': "https://www.sciencefun.org/kindergarten-triva/",
                         'Year 1': "https://www.ecosystemforkids.com/1st-grade-science-games.html",
                         'Year 2': "https://www.ecosystemforkids.com/2nd-grade-science-games.html",
                         'Year 3': "https://www.ecosystemforkids.com/3rd-grade-science-games.html",
                         'Year 4': "https://www.ecosystemforkids.com/4th-grade-science-games.html",
                         'Year 5': "https://www.ecosystemforkids.com/5th-grade-science-games.html",
                         'Year 6': "https://www.ecosystemforkids.com/6-grade-science-games.html",
                         'Year 7': "https://www.ecosystemforkids.com/7th-grade-science-games.html",
                         'Year 8': "https://www.ecosystemforkids.com/8th-grade-science-games.html",
                         'Year 9': "https://www.ecosystemforkids.com/9th-grade-science-games.html",
                         'Year 10': "https://www.studiosity.com/student-resources/practice-tests?year_level=8&subject=show-all#content-filter",
                         'Year 11': "https://www.studiosity.com/student-resources/practice-tests?year_level=9&subject=show-all#content-filter",
                         'Year 12': "https://www.studiosity.com/student-resources/practice-tests?year_level=10&subject=show-all#content-filter"}},
    ]
    
    # List to hold instantiated subject tabs
    subject_tabs = []
    
    # Iterate through each subject's resources and create a SubjectTab
    for subject_info in subjects:
        # Create a SubjectTab using the subject_info
        subject_tab = SubjectTab(tabControl, subject_info)
        
        # Append the created SubjectTab to the subject_tabs list
        subject_tabs.append(subject_tab)
        
        # Add the SubjectTab to the tab control in the GUI
        tabControl.add(subject_tab.frame, text=subject_info["title"])

    # Frame for the image inside the top_frame
    image_frame = tk.Frame(top_frame)
    image_frame.pack(fill="both", expand=True)

    # Add image to the image_frame
    learning_image = Image.open("learning.jpg")
    learning_image = learning_image.resize((400, 300), Image.ANTIALIAS)

    learning_photo = ImageTk.PhotoImage(learning_image)
    learning_label = tk.Label(image_frame, image=learning_photo)
    learning_label.image = learning_photo
    learning_label.pack(side='left', fill="both", expand=True)

    # Configure row weights for main_frame so they expand properly
    main_frame.grid_rowconfigure(0, weight=1)  # top_frame row
    main_frame.grid_rowconfigure(1, weight=1)  # bottom_frame row

    # Initialize ChatBot
    ChatBot(root)

    # Calculate position for centering the window on the screen
    root.update_idletasks()  # This makes sure the window dimensions are correctly computed
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the center position
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2) - 40)

    # Position the window at the center
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    
    root.mainloop()
    
# Initialize the Tkinter window for login
root = Tk()
root.title("Login")

# Create labels for the Name and Password
label_1 = Label(root, text="ID")
label_2 = Label(root, text="Password")

# Create entry widgets to accept user input
entry_1 = Entry(root)
entry_2 = Entry(root, show="*")  # Password masked

# Place the labels and entry widgets using the grid layout
label_1.grid(row=0, sticky=E)
label_2.grid(row=1, sticky=E)
entry_1.grid(row=0, column=1)
entry_2.grid(row=1, column=1)

# Add Checkbutton to toggle password visibility
check_var = BooleanVar()
check_button = Checkbutton(root, text="Show Password", variable=check_var, command=toggle_password)
check_button.grid(row=0, column=2)

# Add Log In button
login_button = Button(root, text="Log In", command=log_in)
login_button.grid(row=1, column=2)
        
# Create and place a Checkbutton for "Keep me logged in" with columnspan set to 3
c = Checkbutton(root, text="Keep me logged in")
c.grid(row=4, columnspan=3)

# Calculate position for centering the window on the screen
root.update_idletasks()  # This makes sure the window dimensions are correctly computed
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the center position
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2) - 100)

# Position the window at the center
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Start the Tkinter event loop
root.mainloop()