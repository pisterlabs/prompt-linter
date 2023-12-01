from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI # Anthropic, AI21

load_dotenv()

template = """Pregunta: {question}

Respuesta: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# question = "Hola, soy Ane. ¿Tú cómo te llamas?"
# question = "¿Cuál es la capital de Francia?"
# question = "¿Recuerdas cómo me llamo?"
# question = "Dame la lista de campeones de la primera división de fútbol desde 2015"
# question = "Dame la lista de campeones de la primera división de fútbol desde 2015, en formato JSON con las claves 'temporada' y 'equipo'"

# llm = Anthropic(temperature=0)
# llm = AI21(temperature=0)
llm = OpenAI(temperature=0)

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
)

# print(llm_chain.prompt)
# print(llm_chain.run(question))

while True:
    question = input("Humano: ")
    answer = llm_chain.run(question)
    print("AI: ", answer)
