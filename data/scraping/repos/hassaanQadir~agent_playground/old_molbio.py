import os
import json
import time
import openai
import pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# Load environment variables
load_dotenv('.env')
# Use the environment variables for the API keys if available
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
# Load agents' data from JSON
with open('agents.json', 'r') as f:
    agentsData = json.load(f)
# Set the OpenAI and Pinecone API keys
openai.api_key = openai_api_key
pinecone.init(api_key=pinecone_api_key, enviroment="us-west1-gcp")
# Name of the index where we vectorized the OpenTrons API
index_name = 'opentronsapi-docs'
index = pinecone.Index(index_name)
def retry_on_error(func, arg, max_attempts=5):
    """
    Retry a function in case of an error. This guards against rate limit errors.
    The function will be retried max_attempts times with a delay of 4 seconds between attempts.
    :param func: The function to retry
    :param arg: The argument to pass to the function
    :param max_attempts: The maximum number of attempts
    :return: The result of the function or a string indicating an API error
    """
    for attempt in range(max_attempts):
        try:
            result = func(arg)
            return result
        except Exception as e:
            if attempt < max_attempts - 1:  # no need to sleep on the last attempt
                print(f"Attempt {attempt + 1} failed. Retrying in 4 seconds.")
                time.sleep(4)
            else:
                print(f"Attempt {attempt + 1} failed. No more attempts left.")
                API_error = "OpenTronsAPI Error"
                return API_error
def askOpenTrons(query):
    """
    Query the OpenTrons API index and return answers
    :param query: The question to ask
    :return: The answer from the API
    """
    embed_model = agentsData[0]['embed_model']
    res = openai.Embedding.create(
        input=["Provide the exact code to perform this step:", query],
        engine=embed_model
    )
    # retrieve from Pinecone
    xq = res['data'][0]['embedding']
    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)
    # get list of retrieved text
    contexts = [item['metadata']['text'] for item in res['matches']]
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    # system message to 'prime' the model
    template = (agentsData[5]['agent5_template'])

    res = openai.ChatCompletion.create(
        model=agentsData[0]['chat_model'],
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": augmented_query}
        ]
    )
    return (res['choices'][0]['message']['content'])
def create_llmchain(agent_id):
    """
    Create a LLMChain for a specific agent by calling on prompts stored in agents.json
    :param agent_id: The ID of the agent
    :return: An instance of LLMChain
    """
    chat = ChatOpenAI(streaming=False, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, openai_api_key=openai_api_key)
    template = agentsData[agent_id]['agent{}_template'.format(agent_id)]
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    example_human = HumanMessagePromptTemplate.from_template(agentsData[agent_id]['agent{}_example1_human'.format(agent_id)])
    example_ai = AIMessagePromptTemplate.from_template(agentsData[agent_id]['agent{}_example1_AI'.format(agent_id)])
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
    return LLMChain(llm=chat, prompt=chat_prompt)
# Create Agents for each Layer, 1 to 4
# The OpenTrons API Agent is created separately in askOpentrons()
chain_1 = create_llmchain(1)
chain_2 = create_llmchain(2)
chain_3 = create_llmchain(3)
chain_4 = create_llmchain(4)
chain_5 = create_llmchain(5)

def main(user_input):
    output_string = ""
    output_1 = chain_1.run(user_input)
    raw_phases = output_1.split('|||')
    phases = [s for s in raw_phases if len(s) >= 10]
    phases = phases[1]
    output_string += "Here are all the phases at once\n\n"
    output_string += "\n".join(phases)
    output_string += "\n"
    for i, phase in enumerate(phases, 1):
        output_string += "\n\nPhase {}\n{}\n".format(i, phase)
        output_2 = chain_2.run(phase)
        raw_steps = output_2.split('|||')
        steps = [s for s in raw_steps if len(s) >= 10]
        steps = steps[1]
        output_string += "Here are all the steps at once for this phase\n\n"
        output_string += "\n".join(steps)
        output_string += "\n"
        for j, step in enumerate(steps, 1):
            output_string += "Step {}\n{}\n".format(j, step)
            output_3 = chain_3.run(step)
            raw_substeps = output_3.split('|||')
            substeps = [s for s in raw_substeps if len(s) >= 10]
            substeps = substeps[1]
            output_string += "Here are all the substeps at once for this step\n\n"
            output_string += "\n".join(substeps)
            output_string += "\n"
            for k, substep in enumerate(substeps, 1):
                output_string += "Substep {}\n{}\n".format(k, substep)
                output_4 = chain_4.run(substep)
                raw_commands = output_4.split('|||')
                commands = [s for s in raw_commands if len(s) >= 5]
                commands = commands[1]
                output_string += "Here are all the commands at once for this substep\n"
                output_string += "\n".join(commands)
                output_string += "\n"
                for l, command in enumerate(commands, 1):
                    output_string += "Line {}\n{}\n".format(l, command)
                    output_string += "Here is the code for this command\n"
                    output_string += retry_on_error(askOpenTrons,command)
                    output_string += "\n\n"
    return output_string
def test(user_input):
    user_input += "Successfully accessed\n"
    user_input += "the molbio.ai\n"
    user_input += retry_on_error(askOpenTrons, "Put 1 ng of DNA stored 50ug/ml into the eppendorf with 100 ul of water")
    return user_input
if __name__ == "__main__":
   #answer = main("Make glow in the dark e. coli")
   answer = test("batman")
   with open('answer.txt', 'w') as f:
    f.write(answer)