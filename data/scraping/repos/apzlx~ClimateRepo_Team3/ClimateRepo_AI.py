import openai
from PyPDF2 import PdfReader
openai.api_key = "your API key here"
climate_categories = "Physical asset risk, Carbon accounting and reporting, supply chain decarbonization, Renewable energy, Carbon pricing, Regulatory environment"

business_disciplines = 'Accounting, Finance, Strategy, Ethics, Marketing, Investment'

model = "gpt-3.5-turbo-0613"

need = input("What are you looking for: review an existing case, or create new content?")
if need == "review an existing case":
    file_name = input("Please enter the file path you want to process:")
    case_text = ''
    reader = PdfReader(file_name)
    # reader = PdfReader("Zara Climate.pdf")
    for page_num in range(0,3):
        page = reader.pages[page_num]
        case_text+=page.extract_text().replace('\n', ' ')

    messages = [
    {"role": "system", "content": f"You are a expert in climate change topics and business. You should categorize a business school case by sustainability issues and by business discipline. For sustainability, only provide one of the six categories in {climate_categories}. For business discipline, only provide one of the six categories in {business_disciplines}."},
    {"role": "user", "content": f"Classify the following input from a case into the categories for climate and for business: {case_text}"},
    ]

    def get_category(messages, model):

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        print(response['choices'][0]['message']['content'])
    
    get_category(messages, model)
elif need == "create new content":
    prompt = input("What are you looking to create?")
    messages = [
    {"role": "system", "content": "You are a business school professor with expertise in sustainability. You should generate a business style case based on given prompt of company name and needs."},
    {"role": "user", "content": f"Create a business case study for {prompt}"},
    ]

    def create_case(messages, model):

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        
        with open(f'{prompt}.txt', 'w') as file:
            file.write(response['choices'][0]['message']['content'])
        print("File successfully created.")
    create_case(messages, model)
