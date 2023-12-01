from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

key = "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"
llm = OpenAI(temperature=0.3, openai_api_key=key)

prompt = PromptTemplate.from_template("what type of doctor should i suggest if, {symptoms}")
chain = LLMChain(llm = llm, prompt=prompt)

doctors = ["practitioner", "physician", "cardiologist", "gastroenterologist", "dermatologist", "neurologist", "orthopedic", "pediatrician", "gynecologist", "obstetrician", "urologist", "nephrologist", "psychiatrist", "dentist", "maxillofacial", "physiotherapist", "ophthalmologist", "allergist", "immunologist", "pulmonologist", "endocrinologist", "ent"]

def predict_doc(symptoms):
    ans = chain.run(symptoms)
    ans = ans.lower()
    print(ans)
    i=1
    for doctor in doctors:
        print(doctor)
        if doctor == "immunologist" or doctor == "obstetrician" or doctor == "maxillofacial":
            i -= 1
        if doctor in ans:
            return i
        i += 1
        
    return 1
    

