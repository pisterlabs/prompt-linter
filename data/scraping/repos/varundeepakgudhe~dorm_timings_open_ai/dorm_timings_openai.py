pip install openai
import openai
openai.api_key = "Your API_KEY"

def get_dorm_info(document, question):
   
    #checks if closing word is in the question to respond for closing timings
    if 'closing' in question.lower():
        prompt = f"Document: {document}\nQuestion: {question}\nAnswer:"
    
    #checks if opening word is in the question to respond for opening timings
    elif 'opening' in question.lower():
        prompt = f"Document: {document}\nQuestion: {question}\nAnswer:"
        
    else:
        return "Appologies, cant understant this question. As an AI i am limited to certain knowledge"
   
    #defining response with only 50 maximum tokens
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=50)

    return response.choices[0].text.strip()

document = "The dorm timings are:\nDorm AbdulKalam: 7:00 AM (opening), 10:59 PM(closing)\nDorm Gangotri: 6:00 AM (opening), 10:30 PM(closing)\nDorm Ganga: 9:00 AM (opening), 9:30 PM(closing)\nDorm Triveni: 7:30 AM (opening), 11:59 PM(closing)\nDorm Gandhi: 8:00 AM (opening), 10:00 PM(closing)\nDorm Yamuna: 7:00 AM (opening), 9:00 PM(closing)"

while True:

    question = input("Hi, please ask me a question about the dorm timings: ")
   
    #breaks the while loop if the question input is exit
    if question.lower()=="exit":
      break
   
    #calling the function get_dorm_info
    else:
      response = get_dorm_info(document, question)
      print(response)

