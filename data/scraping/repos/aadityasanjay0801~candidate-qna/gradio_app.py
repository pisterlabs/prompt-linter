import gradio as gr
import resume_parser as rp
import langchain
# import fitz
# from PIL import Image

def greet(name):
    return "Hello " + name + "!"

# def render_file(file):
#     global N
    
#     # Open the PDF document using fitz
#     doc = fitz.open(file.name)
    
#     # Get the specific page to render
#     page = doc[N]
    
#     # Render the page as a PNG image with a resolution of 300 DPI
#     pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    
#     # Create an Image object from the rendered pixel data
#     image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    
#     # Return the rendered image
#     return image

pages = None

retriever = None

def get_file(file):
  # Process the PDF file here
  return

def process_context_from_pdf(files):
    # file_paths = [file.name for file in files]
    # print(files)
    global pages
    global retriever
    pages = get_pages(files)
    print("PDF Pages exraction Done")
    retriever = create_index(pages)
    print("PDF Indexing Done")
    print("PDF Processing Done")

    
    return 

    # return file_paths
def get_pages(files):
    print("Getting pages")
    pages = rp.load_document(files)
    # print(pages)
    return pages

def create_index(pages):
    print("Creating index")
    index = rp.create_index(pages)
    return index

def retrieve(query, chat_history):
    print("Retrieving")
    response = {}
    # try:
    response = rp.get_retriever(query, retriever)
    # except:
        # print("Error in retrieving")
    # response['result'] = "Sorry, I couldn't find anything in the resume. Please try again."
    chat_history.append((query,response['result']))
    return "",chat_history

with gr.Blocks() as demo:
    
    with gr.Column(scale = 0.9):
        with gr.Row():
            
            file_output = gr.File()
            upload_button = gr.UploadButton("Click to Upload the resume file (PDF)", file_types=["pdf"],
                                            #  file_count="multiple"
            )
        upload_button.upload(process_context_from_pdf, upload_button, file_output)
        
    with gr.Row():
            chatbot = gr.Chatbot(value=[],
                                 layout= "bubble",
                                 label = "Candidate QnA",
                                 show_label = True,
                                elem_id='chatbot')                               
    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter query about candidate and press enter"
            )

        # with gr.Column(scale=0.15):
        #     submit_btn = gr.Button('Submit')

        # with gr.Column(scale=0.15):
        #     btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"]).style()
    txt.submit(retrieve, inputs=[txt,chatbot], outputs=[txt,chatbot])
    
    
demo.launch(debug=True)

    
