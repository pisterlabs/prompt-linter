import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

museum_messages = [
    {
        "role" : "system",
        "content" : (
                        "You are an interactive assistant at a museum that specializes in natural history and science exhibits.\n"
                        "The museum attracts visitors of all ages. Your primary objective should be to enhance the visitor experience\n"
                        "by providing information, answering questions, and engaging in interactive conversations about the exhibits.\n"
                    )
    },
    {
        "role" : "assistant",
        "content" : ("Start each reponse with the phrase: 'Thanks for reaching out. '")
    }
]


while(1):
        
    print("How may I help you?")
    p = input()
    museum_messages.append(
        {
            "role" : "user",
            "content" : p
        }
    )

    try:
        museum_messages_formatted = "".join([f"{msg['role']} : {msg['content']}" for msg in museum_messages])
        response = openai.completions.create(
            model = "text-davinci-003",
            prompt = museum_messages_formatted,
            temperature = 0.1,
            max_tokens = 150,
        )

        response_recieved = response.choices[0].text.strip()
        
        print(response_recieved)
        
        museum_messages.append(
            {
                "role" : "assistant",
                "content" : response_recieved
            }
        )

        print("Do you want me to help you with anything else? y/n")
        f = input()
        if(f=='n'):
            break;

    except  Exception as e:
        print(f"\nAn ERROR recieved from OpenAI's API: {e}\n")


