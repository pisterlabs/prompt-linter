from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def need_agent(prompt):
    llm = AzureChatOpenAI(deployment_name="gpt35-team-3-0301", temperature=0)
    context = SystemMessage(
        content=f"""
        Let's define an agent as an LLM model that can use tools to enhance its output, or to be able to even achieve an output for a given prompt. \
        Let's define a chatbot as an LL model capable of having chatting habilities and having a general knowledge about most things, but isnt able to give much deeper help like an agent. \
        For example, a chatbot is better when the promp wants the LLM to explain a concept. The agent is better to look out for information or make equations. \
        Agents also have some change of crashing due to the nature or intent of the prompt, like in the case of chatting or asking simple conceptual questions, where the chatbot is ALWAYS prefered. \
        Your task is to determine which is LLM model (agent or chatbot) is better to answer a given prompt.
        The input given to you will be given with the following format (the example is delimited by ```):

        ```
        Is the agent the best LLM to answer this prompt?

        Prompt:
        <prompt given by the human>
        ```

        You MUST answer with \"YES.\" or \"NO.\", and nothing else.
        """
    )
    format_prompt = f"""
        Is the agent the best LLM to answer this prompt?

        Prompt:
        {prompt}
    """
    answer = llm(messages=[context, HumanMessage(content=format_prompt)])

    print(f"\nIS AGENT NEEDED? {answer.content}\n")

    return (answer.content == "YES.")