from langchain import PromptTemplate

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/q9jZGwb/wizard.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/DC4Lgdn/human.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

prompt_template_en = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}
Answer:"""

custom_prompt_en = PromptTemplate(
    template=prompt_template_en,
    input_variables=["context", "question"]
)

prompt_template_es = """Utilice las siguientes piezas de contexto para responder la pregunta al final.
Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.
Mantenga su respuesta lo más concisa posible.

{context}

Pregunta: {question}
Respuesta:"""

custom_prompt_es = PromptTemplate(
    template=prompt_template_es,
    input_variables=["context", "question"]
)
