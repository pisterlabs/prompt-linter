from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for patient diagnosis
diagnosis_template = '''Diagnose the medical condition based on the following symptoms:
Symptoms: {symptoms}
Patient Information: {patient_info}'''

diagnosis_prompt = PromptTemplate(
    input_variables=["symptoms", "patient_info"],
    template=diagnosis_template
)

# Format the patient diagnosis prompt
diagnosis_prompt.format(
    symptoms="Fever, headache, cough",
    patient_info="Age: 35, Gender: Male"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
diagnosis_chain = LLMChain(llm=llm, prompt=diagnosis_prompt)

# Run the patient diagnosis chain
diagnosis_chain.run({
    "symptoms": "Fever, headache, cough",
    "patient_info": "Age: 35, Gender: Male"
})
