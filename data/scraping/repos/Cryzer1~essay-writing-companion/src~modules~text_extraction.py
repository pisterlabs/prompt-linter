# Updating the text extraction module to handle the BytesIO object issue with PyPDFLoader from langchain.
# The function will now write the BytesIO object to a temporary file, which will then be used with PyPDFLoader.

def extract_texts_from_files(uploaded_files):
    """Extract text from the provided list of uploaded PDF files using PyPDFLoader."""
    from langchain.document_loaders import PyPDFLoader
    import io
    import tempfile
    
    all_docs = []
    errors = []
    
    for file in uploaded_files:
        try:
            # Write BytesIO object to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            
            # Use PyPDFLoader to load the content from the temporary file
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()
            
            # Append the 'pages' object for later use in the app
            all_docs.append(pages)
            
        except Exception as e:
            errors.append(f"Failed to extract text from {file.name}: {str(e)}")
    
    return all_docs, errors

# This updated function writes the BytesIO object to a temporary file and then uses PyPDFLoader to load and split the PDF.
# It returns 'pages' objects in 'all_docs' and any error messages in 'errors'.
