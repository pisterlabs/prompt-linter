from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for crop disease diagnosis
diagnosis_template = '''Diagnose the disease affecting the crop based on the following symptoms:
Crop: {crop}
Symptoms: {symptoms}'''

diagnosis_prompt = PromptTemplate(
    input_variables=["crop", "symptoms"],
    template=diagnosis_template
)

# Format the crop disease diagnosis prompt
diagnosis_prompt.format(
    crop="Tomatoes",
    symptoms="yellowing leaves, spots on fruits"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
diagnosis_chain = LLMChain(llm=llm, prompt=diagnosis_prompt)

# Run the crop disease diagnosis chain
diagnosis_chain.run({
    "crop": "Tomatoes",
    "symptoms": "yellowing leaves, spots on fruits"
})


#OUTPUT

"""


The disease affecting the crop is likely to be bacterial spot. 
Bacterial spot is caused by a bacterial pathogen, Xanthomonas campestris pv. 
vesicatoria, which is spread by splashing rain or overhead irrigation. 
Symptoms of this disease include yellowing leaves, spots on fruits, and premature fruit drop.
""""
