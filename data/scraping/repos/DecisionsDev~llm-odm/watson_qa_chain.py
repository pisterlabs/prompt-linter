
import os
from dotenv import load_dotenv


from genai.credentials import Credentials
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Check environment variable
def checkEnvironment():
    if not "WATSON_URL" in os.environ:
        print("#Error you should set the environment variable WATSON_URL that correspond to the Watson URL.")
        exit(1)
        
    if not "WATSON_API_KEY" in os.environ:
        print ("#Error you should set the environment variable WATSON_API_KEY. This variable should contain the API Key to connect to watson.")
        exit(1)

    if not "WATSON_PROJECT_ID" in os.environ:
        print ("#Error you should set the environment variable WATSON_PROJECT_ID. This variable should contain the PROJECT_ID  to connect to watson.")
        exit(1)

def create_qa_chain(vectordb: Chroma, qaPrompt: str ):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """
    checkEnvironment()

    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 2000
    }

    credentials = {
        "url": os.environ["WATSON_URL"],
        "apikey": os.environ["WATSON_API_KEY"] #getpass.getpass("Type the api key and return")
    }

    project_id = os.environ["WATSON_PROJECT_ID"]



    for model in ModelTypes:
        print(model)
        
    #model_id = ModelTypes.FLAN_UL2
    model_id = ModelTypes.LLAMA_2_70B_CHAT
    #model_id = ModelTypes.GRANITE_13B_CHAT
#    model_id = ModelTypes.GRANITE_13B_INSTRUCT
   #model_id =ModelTypes.GPT_NEOX


    llm = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(llm.to_langchain(), 
                                                    vectordb.as_retriever(),
                                                    chain_type="stuff",
                                                    combine_docs_chain_kwargs={"prompt": qaPrompt},                
                                                    memory=memory)
    return qa_chain

