
def main():
    import openai
    import streamlit as st
    import PyPDF2
    from PyPDF2 import PdfReader
    import io

    # Set API key as a secret
    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()
    openai.api_key = api_key

    # Set up Streamlit app 
    st.title("PDF Summary")
    file = st.file_uploader("Upload a PDF file", type="pdf")

    # Generate summary if file is uploaded
    if file is not None:
        # Read PDF content
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        num_pages = len(pdf_reader.pages)
        text = ''
        for page in range(num_pages):
            page_obj = pdf_reader.pages[page]
            text += page_obj.extract_text()

        # Generate summary using OpenAI API
        prompt = f"Summarize this in 350 words, including Who is the client's name? Who is the name in medical treatment? +\
        Are they the same names? If different names, indicate it as suspicious: {text}"        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].text.strip()

        # Display summary to user
        st.write("Here is a summary of the PDF content:")
        st.write(answer)

if __name__ == '__main__':
    main()
