import openai
import gradio
from services.analysys import validateQuestionforWord

openai.api_key_path= "credencialOpenai.txt"

"""in role user, the content is the question, in role system, the content is the answer"""
"""messages=[
        {"role": "system",
         "content": "Tú eres un psícologo especialista en test vocacionales, harás preguntas y responderás para dar a conocer a que carrera se puede dedicar."
        }
]"""
    
"""This function is to answer the question of the user, the question is the parameter of the function."""
def answerQuestion(nameUser, questionUser):
    messages=[
        {"role": "system",
         "content": "Tú eres un psícologo especialista en test vocacionales, harás preguntas y responderás para dar a conocer a que carrera se puede dedicar."
        }
    ]
    sms = validateQuestionforWord(questionUser)
    if sms == None:
        messages.append({"role": "user","content": questionUser})
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages
        )
        replay = response["choices"][0]["message"]["content"]
    else:
        replay = sms
    messages.append({"role": "assistant","content": replay})
    welcome = f"Bienvenido(a) {nameUser}, a tu asesor personalizado para conocer tu perfil vocacional. ¡Empecemos!"

    return welcome, replay
   

demo = gradio.Interface(
    fn=answerQuestion, 
    inputs= [gradio.inputs.Textbox(lines=1, placeholder="ColoqueSu nombre aquí..."),gradio.Textbox(lines=5, placeholder="Escriba su pregunta aquí...")],
    outputs=["textbox", "textbox"],
    title="Smart Test Careers", 
    description="Este es un chatbot que te ayudará a encontrar tu vocación profesional."
)
demo.launch(share=True)