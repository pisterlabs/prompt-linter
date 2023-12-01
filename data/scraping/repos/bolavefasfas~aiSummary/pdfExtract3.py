import openai
from PyPDF2 import PdfFileReader

# Create an instance of the GPT-3 model
openai.api_key = "your api key goes here"

# open the PDF file
pdfFile = open('target.pdf', 'rb')

# create PDFFileReader object to read the file
pdfReader = PdfFileReader(pdfFile)

print("Printing the document info: " + str(pdfReader.getDocumentInfo()))
print("- - - - - - - - - - - - - - - - - - - -")
print("Number of Pages: " + str(pdfReader.getNumPages()))

numOfPages = pdfReader.getNumPages()

for i in range(0, numOfPages):
    pageObj = pdfReader.getPage(i)
    targetText = pageObj.extractText(i)
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Write a summary for this in the style of the New Yorker: " + targetText,
    max_tokens=800,
    n=1,
    temperature=0.5,
)
    # Print the generated summary
    print(response["choices"][0]["text"])

    # Open a text file for writing
    with open('summaries.txt', 'a') as f:
        # Write the list of bullet point summaries to the file
        f.writelines(response["choices"][0]["text"])

# close the PDF file object
pdfFile.close()
