
# Credit 🙏: I just used the example from langchain docs and it works quite well: https://python.langchain.com/en/latest/use_cases/question_answering.html
# Note 2: The Arxiv -> PDF logic is a bit messy, I'm sure it can be done better
# Note 3: Please install the following:

# To run:

# Save this in a `app.py`
# pip install arxiv PyPDF2 langchain chromadb

# The chat feature was shipped in H2O nightly this week, we will need to install from nightly link:
# pip install https://github.com/h2oai/wave/releases/download/nightly/h2o_wave-nightly-py3-none-manylinux1_x86_64.whl

# Before running the app, make sure you export your openai key as an environment variable:
# export OPENAI_API_KEY = "YOUR_KEY_GOES_HERE"
# Run the following command and navigate to localhost:10101/demo, Note: Don't put app.py, just app 👇
# wave run app

# You will need to put the paper number and then ask questions
# Shout at me for errors: https://twitter.com/bhutanisanyam1


import arxiv
import PyPDF2
import io
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from h2o_wave import main, app, Q, ui, data

MAX_MESSAGES = 500  # Maximum number of messages to store in the chat window
index = None  # Initialize the global index variable

# Function to convert PDF to text
def pdf_to_txt(file_path):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        text = ''

        # Extract text from each page of the PDF
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()

    return text

# Function to prepare the index for querying
def prepare_index(txt_file_path):
    global index
    loader = TextLoader(txt_file_path)
    index = VectorstoreIndexCreator().from_loaders([loader])

# Main app function that handles the UI and logic
@app('/demo')
async def serve(q: Q):
    global index

    # Initialize the UI if it hasn't been initialized yet
    if not q.client.initialized:
        # Set up the UI theme
        q.page['meta'] = ui.meta_card(
            box='',
            themes=[
                ui.theme(
                    name='dracula',
                    primary='#50fa7b',
                    text='#f8f8f2',
                    card='#282a36',
                    page='#282a36',
                )
            ],
            theme='dracula'
        )

        # Set up the chat title
        q.page['chat_title'] = ui.form_card(
        box='3 1 9 1',
        items=[
        ui.text('<h1 style="text-align: center; color: #50fa7b;">ArXiv Chat: Chat with Latest Papers</h1>'),
        ]
        )

        # Set up the chat window
        cyclic_buffer = data(fields='msg fromUser', size=-MAX_MESSAGES)
        q.page['chat'] = ui.chatbot_card(box='3 2 9 8', data=cyclic_buffer, name='chat')

        # Set up the input form for Arxiv ID and query
        q.page['input'] = ui.form_card(box='3 8 9 3', items=[
            ui.textbox(name='arxiv_id', label='Arxiv ID', placeholder='Enter Arxiv ID (e.g. 2303.17580)', disabled=q.client.arxiv_id_set),
            ui.textbox(name='query', label='Query', placeholder='Enter your query'),
            ui.buttons(items=[ui.button(name='submit', label='Submit', primary=True)]),
        ])

        # Initialize client variables
        q.client.arxiv_id = None
        q.client.arxiv_id_set = False
        q.client.initialized = True
        await q.page.save()

    # Handle the submit button click
    if q.args.submit:
        user_query = q.args.query

        # If Arxiv ID is not set, set it
        if not q.client.arxiv_id:
            q.client.arxiv_id = q.args.arxiv_id

            # Check if Arxiv ID is valid
            if not q.client.arxiv_id:
                q.page['chat'].data[-1] = ["Please enter a valid Arxiv ID.", False]
                await q.page.save()
                return

            # Download the paper and convert it to text
            search = arxiv.Search(id_list=[q.client.arxiv_id])
            paper = next(search.results())
            paper.download_pdf(filename="downloaded-paper.pdf")

            pdf_file_path = 'downloaded-paper.pdf'
            txt_file_path = 'converted-paper.txt'

            text = pdf_to_txt(pdf_file_path)

            # Save the text to a file
            with io.open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Prepare the index for querying
            prepare_index(txt_file_path)

        # Query the index if it exists
        if index:
            response = index.query(user_query)
            # Append user message
            q.page['chat'].data[-1] = [f"User: {user_query}", True]
            # Append bot response
            q.page['chat'].data[-1] = [f"Bot: {response}", False]

            q.client.arxiv_id_set = True

        await q.page.save()

# Main entry point
if __name__ == '__main__':
    main()