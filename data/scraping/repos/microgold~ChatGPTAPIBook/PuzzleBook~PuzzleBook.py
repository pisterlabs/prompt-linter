import shutil
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportLabImage, PageBreak, Spacer, Table, TableStyle
from io import BytesIO
import os
import random
import string
import time
import tkinter as tk
from tkinter import ttk
from dotenv import load_dotenv
import openai
import requests
from PIL import Image, ImageTk, ImageDraw, ImageFont
from PuzzleBoardCreator import PuzzleBoardCreator

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportLabImage, PageBreak, Frame, PageTemplate

from reportlab.lib.styles import getSampleStyleSheet
from tkinter import messagebox

pil_image_path = "c:\\temp\\temp__puzzlebook_image.png"
puzzle_image_path = 'c:\\temp\\alphabet_grid.png'

puzzle_board_creator = PuzzleBoardCreator()

generated_image = None

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
openai.api_key = key


def set_wait_cursor():
    submit_btn.config(cursor="watch")
    app.update_idletasks()  # Force an immediate update of the window
    time.sleep(2)  # Simulate some long operation


def set_normal_cursor():
    submit_btn.config(cursor="")


def header(canvas, doc, content):
    canvas.saveState()
    w, h = content.wrap(doc.width, doc.topMargin)
    content.drawOn(canvas, doc.leftMargin, doc.height +
                   doc.bottomMargin + doc.topMargin - h)
    canvas.restoreState()


def footer(canvas, doc, content):
    canvas.saveState()
    w, h = content.wrap(doc.width, doc.bottomMargin)
    content.drawOn(canvas, doc.leftMargin, h)
    canvas.restoreState()


def dynamic_header_and_footer(canvas, doc):
    print('calling dynamic header and footer')
    # Dynamically determine or generate header_content and footer_content here
    styles = getSampleStyleSheet()
    header_content = Paragraph(
        f"Dynamic Header for Page {doc.page}", styles['Normal'])
    footer_content = Paragraph(
        f"Dynamic Footer for Page {doc.page}", styles['Normal'])

    header(canvas, doc, header_content)
    footer(canvas, doc, footer_content)


def create_book(puzzle_words_list, theme_images_list, puzzle_images_list, puzzle_descriptions):

    try:
        print("creating book...")
        if not all([puzzle_words_list, theme_images_list, puzzle_images_list, puzzle_descriptions]):
            messagebox.showerror(
                "Error", "Please provide non-empty lists of puzzle words, theme images, puzzle images, and puzzle descriptions!")
            return

        if not len(puzzle_words_list) == len(theme_images_list) == len(puzzle_images_list) == len(puzzle_descriptions):
            messagebox.showerror(
                "Error", "All input lists must be of the same length!")
            return

        custom_page_size = (6*72, 9*72)
        custom_margins = 0.5*72
        doc = SimpleDocTemplate("output.pdf",
                                pagesize=custom_page_size,
                                topMargin=custom_margins,
                                bottomMargin=custom_margins,
                                leftMargin=custom_margins,
                                rightMargin=custom_margins)
        styles = getSampleStyleSheet()
        contents = []

        headline_style = styles['Heading1']
        normal_style = styles['Normal']

        for i in range(len(puzzle_words_list)):
            header_data = [[Paragraph(f"{puzzle_descriptions[i]}", styles['Normal']),
                            Paragraph(f"{i + 1}", styles['Normal'])]]

            # Adjust colWidths as needed
            # Create a table for the header with spacer on the left, topic in the middle, and page number on the right
            margin_offset = 1*72
            header_data = [['', Paragraph(f"Topic: {puzzle_descriptions[i]}", styles['Normal']),
                            Paragraph(f"Page {i + 1}", styles['Normal'])]]

            header_table = Table(header_data, colWidths=[
                                 margin_offset, 4*72, 2*72])
            header_table.setStyle(TableStyle([
                ('ALIGN', (1, 0), (2, 0), 'RIGHT'),
                # ... (other styling)
            ]))

            contents.append(header_table)
            # Add some space between header and content
            contents.append(Spacer(1, .5*72))

            img1 = ReportLabImage(
                theme_images_list[i], width=1.5*inch, height=1.5*inch)
            contents.append(img1)

            contents.append(Spacer(1, .25*72))

            img2 = ReportLabImage(
                puzzle_images_list[i], width=4*inch, height=4*inch)
            contents.append(img2)

            contents.append(Spacer(1, .25*72))

            puzzle_word_text = '<br/>' + puzzle_words_list[i]
            puzzle_word_text = puzzle_word_text.replace('\n', '<br/><br/>')
            paragraph = Paragraph(puzzle_word_text, normal_style)
            contents.append(paragraph)

            if i < len(puzzle_words_list) - 1:
                contents.append(PageBreak())

        doc.build(contents)
        messagebox.showinfo("PDF Created", "PDF created successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error creating PDF: {e}")

# Example usage:
# create_book(
#     ["word1, word2", "word3, word4"],
#     ["path/to/theme_image1.png", "path/to/theme_image2.png"],
#     ["path/to/puzzle_image1.png", "path/to/puzzle_image2.png"],
#     ["Puzzle Description 1", "Puzzle Description 2"]
# )


def create_pdf(puzzle_word_text):
    if not puzzle_word_text:
        messagebox.showerror("Error", "Please generate the dialog first!")
        return

    theme = combo1.get()

    # 3. Create a PDF with both the extracted image and some text
    doc = SimpleDocTemplate("output.pdf", pagesize=letter)

    # Create the contents list for the PDF
    contents = []

    styles = getSampleStyleSheet()
    headline_style = styles['Heading1']
    headline = Paragraph(theme.capitalize(), headline_style)
    contents.append(headline)

    # Add the extracted image
    # Adjust width and height as needed
    img1 = ReportLabImage(pil_image_path, width=2.5*inch, height=2.5*inch)
    contents.append(img1)
    img2 = ReportLabImage(puzzle_image_path, width=5*inch, height=5*inch)
    contents.append(img2)

    # Add some text
    puzzle_word_text = '<br/>' + puzzle_word_text
    puzzle_word_text = puzzle_word_text.replace('\n', '<br/><br/>')
    styles = getSampleStyleSheet()
    paragraph = Paragraph(puzzle_word_text, styles['Normal'])
    contents.append(paragraph)

    # Build the PDF
    doc.build(contents)

    # message box saying we finished generating the PDF
    messagebox.showinfo("PDF Created", "PDF created successfully!")


def generate_image(theme):
    response = openai.Image.create(
        model="image-alpha-001",
        prompt=f"cartoon image of {theme}",
        n=1,  # Number of images to generate
        size="256x256",  # Size of the generated image
        response_format="url"  # Format in which the image will be received
    )

    image_url = response.data[0]["url"]
    print(image_url)
    return image_url


def update_label_with_new_image(label, photo):
    # Assuming `label` is a global variable
    # photo = get_photo_from_url(new_image_url)
    label.config(image=photo)
    label.image = photo  # Keep a reference to avoid garbage collection


def display_image_from_url(image_holder, url):
    # Fetch the image from the URL
    response = requests.get(url)
    image_data = BytesIO(response.content)

    # Open and display the image using PIL and tkinter
    image = Image.open(image_data)
    image.resize((200, 200), Image.ANTIALIAS)

    # Save the image as a PNG
    image.save(pil_image_path, "PNG")

    photo = ImageTk.PhotoImage(image)

    update_label_with_new_image(image_holder, photo)
    return image


def display_image_from_path(image_holder, path):
    # Fetch the image from the file

    # open image from path 'c:\\temp\\alphabet_grid.png'
    image = Image.open(path)

    photo = ImageTk.PhotoImage(image)

    update_label_with_new_image(image_holder, photo)
    return image


def create_grid_of_letters_image(letters):
    # Set image size, background color, and font size
    img_size = (530, 530)
    background_color = (255, 255, 255)  # white
    font_size = 30

    # Create a new image with white background
    img = Image.new('RGB', img_size, background_color)
    d = ImageDraw.Draw(img)

    # Load a truetype or OpenType font file, and set the font size
    try:
        fnt = ImageFont.truetype(
            'C:\\Windows\\Fonts\\Cour.ttf', font_size)
    except IOError:
        print('Font not found, using default font.')
        fnt = ImageFont.load_default()

    # Generate the 13 by 13 grid of letters

    for i in range(13):
        for j in range(13):
            letter = letters[i][j]  # Cycle through the alphabet
            # Adjust position for each letter
            position = (j * (font_size + 10) + 10, i * (font_size + 10) + 10)
            # Draw letter with black color
            d.text(position, letter, font=fnt, fill=(0, 0, 0))

    # Save the image
    img.save(puzzle_image_path)


def clean_words(words):

    # remove any words that repeat
    words = list(dict.fromkeys(words))

    # choose only words that are 10 characters or less when punctuation is stripped
    # and spaces removed
    clean_words = []
    for word in words:
        word = word.upper().replace(" ", "")
        # remove any punctuation
        word = word.translate(str.maketrans('', '', string.punctuation))
        if (len(word) <= 10):
            clean_words.append(word)

    # narrow down words to only a list of 10
    if len(clean_words) > 10:
        clean_words = random.sample(clean_words, 10)

    # sort the words by size with largest first
    clean_words.sort(key=len, reverse=True)
    return clean_words


def copy_image(src_path, suffix):
    try:
        dst_path = src_path+'_'+suffix
        shutil.copy2(src_path, dst_path)
        print(f'Successfully copied {src_path} to {dst_path}')
        return dst_path
    except FileNotFoundError:
        print(f'The file at {src_path} was not found.')
    except PermissionError:
        print(f'Permission denied. Unable to write to {dst_path}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def batch_submit():
    puzzle_words_list = []
    theme_images_list = []
    puzzle_images_list = []
    puzzle_descriptions = []

    set_wait_cursor()
    theme = combo1.get()

    prompt = f"Create a comma delimited list of 40 words having to do with the theme {theme}. None of the words in the list should repeat\n"
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )

    print(prompt)

    # retrieve the list of words created by ChatGPT
    chatGPTAnswer = response["choices"][0]["message"]["content"]
    print(chatGPTAnswer)
    # split the comma delimited list of words into a list
    topics = chatGPTAnswer.split(',')

    topics = clean_words(topics)  # pick out a list of 10 viable words
    print(topics)

    # now create a list of words from each of those words
    for topic in topics:
        print(topic)
        # save puzzle description
        puzzle_descriptions.append(topic)

        prompt = f"Create a comma delimited list of 40 words having to do with the theme {topic}. None of the words in the list should repeat\n"
        messages = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )

        print(prompt)

        # retrieve the list of words created by ChatGPT
        chatGPTAnswer = response["choices"][0]["message"]["content"]
        print(chatGPTAnswer)
        # split the comma delimited list of words into a list
        words = chatGPTAnswer.split(',')
        words = clean_words(words)  # pick out a list of 10 viable words
        print(words)

        # create word search puzzle array from words
        (board, words_to_remove) = puzzle_board_creator.create_word_search(words)
        # remove words that could not be placed
        words = [word for word in words if word not in words_to_remove]
        puzzle_words_list.append(', '.join(words))
        # show the board on the console
        puzzle_board_creator.display_board(board)
        label_puzzle_words.config(text=', '.join(words))
        # make result_text scrollable

        result_text.config(state="normal")
        result_text.delete(1.0, tk.END)  # Clear any previous results
        result_text.insert(tk.END, chatGPTAnswer)
        result_text.config(state="disabled")

        # generates a cartoon image of the theme
        image_url = generate_image(topic)
        # creates a grid of letters into an image for the puzzle
        create_grid_of_letters_image(board)
        display_image_from_url(image_holder, image_url)
        dest_theme_image_path = copy_image(pil_image_path, topic)
        theme_images_list.append(dest_theme_image_path)

        display_image_from_path(puzzle_holder, puzzle_image_path)
        dest_puzzle_image_path = copy_image(puzzle_image_path, topic)
        puzzle_images_list.append(dest_puzzle_image_path)

        puzzle_holder.config(width=600, height=600)
        set_normal_cursor()

    create_book(puzzle_words_list, theme_images_list,
                puzzle_images_list, puzzle_descriptions)


def submit():
    set_wait_cursor()
    theme = combo1.get()

    prompt = f"Create a comma delimited list of 40 words having to do with the theme {theme}. None of the words in the list should repeat\n"
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )

    print(prompt)

    # retrieve the list of words created by ChatGPT
    chatGPTAnswer = response["choices"][0]["message"]["content"]
    print(chatGPTAnswer)
    # split the comma delimited list of words into a list
    words = chatGPTAnswer.split(',')
    words = clean_words(words)  # pick out a list of 10 viable words
    print(words)
    # create word search puzzle array from words
    (board, words_to_remove) = puzzle_board_creator.create_word_search(words)
    # remove words that could not be placed
    words = [word for word in words if word not in words_to_remove]
    # show the board on the console
    puzzle_board_creator.display_board(board)
    label_puzzle_words.config(text=', '.join(words))
    # make result_text scrollable

    result_text.config(state="normal")
    result_text.delete(1.0, tk.END)  # Clear any previous results
    result_text.insert(tk.END, chatGPTAnswer)
    result_text.config(state="disabled")

    # generates a cartoon image of the theme
    image_url = generate_image(theme)
    # creates a grid of letters into an image for the puzzle
    create_grid_of_letters_image(board)
    display_image_from_url(image_holder, image_url)
    display_image_from_path(puzzle_holder, puzzle_image_path)
    puzzle_holder.config(width=600, height=600)
    set_normal_cursor()


app = tk.Tk()
app.title("Word Puzzle Book")


# Label and ComboBox for the first animal
label1 = ttk.Label(app, text="Select a Word Puzzle Theme:")
label1.grid(column=0, row=0, padx=10, pady=5)
combo1 = ttk.Combobox(
    app, values=["Holidays", "Science", "Travel", "AI", "Cars", "Food", "Entertainment", "Sports", "Space", "Work", "School", "Animals", "Nature", "Art", "Music", "Movies", "Books", "History", "Math", "Geography", "Weather", "Fashion", "Health", "Family", "Money", "Politics", "Religion", "Technology", "Games", "Business", "Crime", "Law", "Medicine", "Psychology", "Language", "Culture", "Relationships", "Social Media", "News", "Shopping", "Transportation", "Architecture", "Design", "Gardening", "Hobbies", "Humor", "Literature", "Philosophy", "Photography", "Writing", "Other"])
combo1.grid(column=1, row=0, padx=10, pady=5)
combo1.set("Holidays")


# Button to submit the details
submit_btn = ttk.Button(app, text="Submit", command=submit)
submit_btn.grid(column=0, row=3, padx=10, pady=20)

# Button to submit the details
create_book_btn = ttk.Button(app, text="Create Book", command=batch_submit)
create_book_btn.grid(column=2, row=3, padx=10, pady=20)

# make it scrollable
# Create a Scrollbar widget
scrollbar = tk.Scrollbar(app)
scrollbar.grid(row=4, column=3, sticky='ns')

# Text widget to display results
result_text = tk.Text(app, width=50, height=10,
                      wrap=tk.WORD, yscrollcommand=scrollbar.set)
result_text.grid(column=0, row=4, rowspan=1, columnspan=4, padx=10, pady=10)
result_text.config(state="disabled")
# result_text.pack(expand=True, fill=tk.BOTH)

image_holder = tk.Label(app)
image_holder.grid(column=0, row=5, columnspan=2, padx=10, pady=10)

puzzle_holder = tk.Label(app)
puzzle_holder.grid(column=5, row=0, rowspan=7,  padx=2,
                   pady=2)

label_key_title = ttk.Label(app, text="Puzzle Words")
label_key_title.grid(column=5, row=6, padx=10, pady=5)

label_puzzle_words = ttk.Label(app, text="")
label_puzzle_words.grid(column=5, row=7, padx=10, pady=10)

# Button to submit the details
create_pdf_btn = ttk.Button(
    app, text="Create Pdf", command=lambda: create_pdf(label_puzzle_words['text']))
create_pdf_btn.grid(column=1, row=3, padx=10, pady=20)

scrollbar.config(command=result_text.yview)


# Link the scrollbar to the text widget (so the scrollbar knows how to scroll the text widget)


app.mainloop()
