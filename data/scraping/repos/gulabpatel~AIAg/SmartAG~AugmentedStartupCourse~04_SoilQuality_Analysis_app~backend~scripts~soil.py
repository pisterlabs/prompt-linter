from xgboost import XGBClassifier
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import numpy as np

loaded_model = XGBClassifier()
loaded_model.load_model("assets/xgboost_model_soil.bin")


def get_data_JSON(relevant_data, loaded_model):
    
    keys = ["N - Ratio of Nitrogen (NH4+) content in soil",
            "P - Ratio of Phosphorous (P) content in soil",
            "K - Ratio of Potassium (K) content in soil",
            "ph - Soil acidity (pH)",
            "ec - Electrical conductivity",
            "oc - Organic carbon",
            "S - Sulfur (S)",
            "zn - Zinc (Zn)",
            "fe - Iron (Fe)",
            "cu - Copper (Cu)",
            "Mn - Manganese (Mn)",
            "B - Boron (B)"]

    output = loaded_model.predict(np.array(relevant_data).reshape(1, -1))

    if output[0] == 0:
        status = "Less fertile"
    elif output[0] == 1:
        status = "Fertile"
    else:
        status = "Highly fertile"

    relevant_data.append(status)
    keys.append("Soil Fertility Status")

    data_dict = dict(zip(keys, relevant_data))

    return data_dict
    


OPENAI_API_KEY='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'


PROMPT_TEMPLATE1="""

You are a Soil Quality Expert. Using the given data on soil health and its associated variables, provide feedback to the user. Here are the details you'll be provided with:

Health Status: This could be "Less fertile", "Fertile", or "Highly fertile".
Variables associated with soil health:
N - Ratio of Nitrogen (NH4+) content in soil
P - Ratio of Phosphorous (P) content in soil
K - Ratio of Potassium (K) content in soil
ph - Soil acidity (pH)
ec - Electrical conductivity
oc - Organic carbon
S - Sulfur (S)
zn - Zinc (Zn)
fe - Iron (Fe)
cu - Copper (Cu)
Mn - Manganese (Mn)
B - Boron (B)
Based on the health status and the variables provided, please provide feedback as follows:

Highly fertile:

Congratulate the user with an excitement-filled message. Mention how the soil is perfect for crops and plantations.
Suggest practices they should use to maintain the fertility of their soil.
Fertile:

Identify which of the provided indicators make the soil quality good.
Mention the weaker factors and provide suggestions on how they can be improved.
Less fertile:

Point out which factors are causing the soil to be less fertile.
Offer solutions and practices on how the soil quality can be improved.


Now


Following is a JSON containing the health status and variables:

   {data_JSON}

"""

PROMPT_TEMPLATE2="""

You are a Soil Quality Expert. Using the given data on soil health and its associated variables, provide feedback to the user. Here are the details you'll be provided with:

Health Status: This could be "Less fertile", "Fertile", or "Highly fertile".
Variables associated with soil health:
N - Ratio of Nitrogen (NH4+) content in soil
P - Ratio of Phosphorous (P) content in soil
K - Ratio of Potassium (K) content in soil
ph - Soil acidity (pH)
ec - Electrical conductivity
oc - Organic carbon
S - Sulfur (S)
zn - Zinc (Zn)
fe - Iron (Fe)
cu - Copper (Cu)
Mn - Manganese (Mn)
B - Boron (B)
Based on the health status and the variables provided, please provide exper feedback
 based on the chat history and user message as follows:

Following is the conversation history between user and expert(You):

   {user_history}

Following is the user message:
  {user_message}

"""

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2, openai_api_key=OPENAI_API_KEY)

PROMPT1 = PromptTemplate(template=PROMPT_TEMPLATE1, input_variables=['data_JSON'])
PROMPT2 = PromptTemplate(template=PROMPT_TEMPLATE2, input_variables=['user_history', 'user_message'])

chain1 = LLMChain(llm=llm, prompt=PROMPT1)
chain2 = LLMChain(llm=llm, prompt=PROMPT2)