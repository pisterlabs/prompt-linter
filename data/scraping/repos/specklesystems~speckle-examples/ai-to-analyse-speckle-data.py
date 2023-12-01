import os, json, glob, re, requests
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Flask, request
from flask_executor import Executor
from PyPDF2 import PdfReader
import openai
import weaviate

from specklepy.api.client import SpeckleClient
from specklepy.objects import Base
from specklepy.transports.server import ServerTransport
from specklepy.api import operations

from langchain.llms import OpenAI
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.text_splitter import NLTKTextSplitter

### This allows for secrets to be set in a .env file adjacent to the script
from dotenv import load_dotenv

load_dotenv()


### ENVIROMENT ###

# should match the webhook setup in speckle admin
flask_url = "0.0.0.0" # change to whatever your server/serverless IP is
flask_port = 5000 # 5000 is the default port for flask
flask_path = "/webhook"

weaviate_url = "http://localhost:8080/" # change to whatever your weaviate server is
speckle_url = "https://speckle.xyz/" # change to whatever your speckle server is

# for legacy conversation history stuffing
openai.api_key = os.getenv["OPENAI_API_KEY"]

### WEAVIATE ###
api_key = os.getenv["OPENAI_API_KEY"]
client = weaviate.Client(
    url=weaviate_url, additional_headers={"X-OpenAI-API-Key": api_key}
)


def weaviate_document(
    client: weaviate.Client, text: str, filename: str
) -> Dict[str, str]:
    """
    This function uploads a document to Weaviate.

    Parameters:
    client (weaviate.Client): The Weaviate client to use for the upload.
    text (str): The text of the document to upload.
    filename (str): The filename of the document to upload.

    Returns:
    Dict[str, str]: A dictionary containing the Weaviate ID of the uploaded document.
    """
    data_object = {"text": text, "filename": filename}
    weaviate_id = client.data_object.create(data_object, class_name="Documents")
    return weaviate_id


### SPECKLE ###

speckle_host = os.getenv["SPECKLE_HOST"]
speckle_token = os.getenv["SPECKLE_TOKEN"]


def extract_comment_data(payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extracts comment data from the payload.

    Args:
        payload (dict): The payload from the webhook.

    Returns:
        tuple: A tuple containing the stream ID,
               comment text, and thread ID.
    """
    stream_id = payload["streamId"]
    activity_message = payload["activityMessage"]
    thread_id = activity_message.split(": ")[1].split(" ")[0]
    comment_text = payload["event"]["data"]["input"]["text"]["content"][0]["content"][
        0
    ]["text"]
    comment_text = comment_text.strip()

    return stream_id, comment_text, thread_id


def extract_reply_data(payload):
    stream_id = payload["streamId"]
    user_id = payload["userId"]

    activity_message = payload["activityMessage"]
    comment_id = (
        activity_message.split("#")[-1].split(" ")[0]
        if "#" in activity_message
        else None
    )

    if "parentComment" in payload["event"]["data"]["input"]:
        thread_id = payload["event"]["data"]["input"]["parentComment"]
        text_data = payload["event"]["data"]["input"]["text"]["content"][0]["content"][
            0
        ]["text"]
    else:
        thread_id = payload["event"]["data"]["input"]["threadId"]
        text_data = payload["event"]["data"]["input"]["content"]["doc"]["content"][0][
            "content"
        ][0]["text"]

    user_question = text_data.strip()

    return stream_id, user_id, user_question, thread_id, comment_id


def get_speckle_client(host: str, token: str) -> SpeckleClient:
    """
    Authenticates and returns a Speckle client.

    This function takes a host and a token, and uses them to authenticate a Speckle client.

    Parameters:
    host (str): The host address for the Speckle server.
    token (str): The personal access token for authentication.

    Returns:
    SpeckleClient: An authenticated Speckle client.
    """
    client = SpeckleClient(host=host)
    client.authenticate_with_token(token)
    return client


def get_object_data(
    client: SpeckleClient, stream_id: str, object_id: str
) -> Dict[str, Any]:
    """
    Retrieves object data from the Speckle Server.

    The function fetches an object from the Speckle server based on its id and the stream id.
    It then processes the object, extracting key information such as type and family, as well as any
    additional parameters depending on the type of object.

    Parameters:
    client: (SpeckleClient) A client instance used to connect to the Speckle server
    stream_id (str): The id of the stream from which to fetch the object.
    object_id (str): The id of the object to fetch.

    Returns:
    Dict[str, Any]: A dictionary containing key information about the fetched object.
    """
    transport = ServerTransport(client=client, stream_id=stream_id)
    original_object: Any = operations.receive(
        obj_id=object_id, remote_transport=transport
    )
    result_dict: Dict[str, Any] = {}

    if (
        original_object.speckle_type
        == "Objects.Other.Instance:Objects.Other.Revit.RevitInstance"
    ):
        definition_object: Any = original_object["definition"]
    else:
        definition_object: Any = original_object

    result_dict["type"] = definition_object.type
    result_dict["family"] = definition_object.family
    result_dict.update(get_object_parameters(definition_object))

    return result_dict


def get_object_parameters(obj: Base) -> Dict[str, Any]:
    """
    Extracts the parameters of a given object.

    This function is specific to Revit and fetches dynamic parameters from the object.
    Each parameter's name and value are then stored in a dictionary for easy access.

    Parameters:
    obj: (Base) The Speckle object from which to extract parameters

    Returns:
    Dict[str, Any]: A dictionary mapping parameter names to their values.
    """
    parameters_data = obj["parameters"]
    parameters = parameters_data.get_dynamic_member_names()

    result_dict: Dict[str, Any] = {
        parameters_data[parameter]["name"]: parameters_data[parameter]["value"]
        for parameter in parameters
    }

    return result_dict


def extract_and_transform_thread(
    raw_thread: Dict[str, Any], thread_id: str
) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Extracts and transforms a thread from raw thread data.

    Args:
        raw_thread (dict): The raw thread data.
        thread_id (str): The ID of the thread to extract.

    Returns:
        tuple: A tuple containing a list of conversation strings and the object ID, or (None, None) if the thread is not found.
    """
    threads = raw_thread["project"]["commentThreads"]["items"]
    for thread in threads:
        if thread["id"] == thread_id:
            # Extract object_id and question
            object_id, question = thread["rawText"].split(" ", 1)
            conversation = [f"Question: {question}"]
            # Sort replies by createdAt
            replies = sorted(thread["replies"]["items"], key=lambda x: x["createdAt"])
            for reply in replies:
                conversation.append(
                    f"Answer: {reply['rawText']}"
                    if reply["authorId"] == "3952b2a678"
                    else f"Question: {reply['rawText']}"
                )
            return conversation, object_id
    return None, None


def speckle_graphql_query(
    query: str, variables: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Sends a GraphQL query to the Speckle server and returns the response.

    Args:
        query (str): The GraphQL query.
        variables (dict, optional): The variables for the GraphQL query. Defaults to None.

    Returns:
        dict: The response data if the request is successful, None otherwise.
    """
    url = f"{speckle_url}graphql"
    payload = {"query": query, "variables": variables}
    token = os.environ["SPECKLE_TOKEN"]
    headers = {"Authorization": token, "Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    return response.json() if response.status_code == 200 else None


def speckle_reply(thread_id: str, reply_text: str) -> Optional[Dict[str, Any]]:
    """
    Posts a reply to a comment thread on the Speckle server.

    Args:
        thread_id (str): The ID of the comment thread.
        reply_text (str): The content of the reply.

    Returns:
        dict: The server's response data if successful, None otherwise.
    """
    mutation = """
        mutation Mutation($input: CreateCommentReplyInput!) {
            commentMutations {
                reply(input: $input) {
                    id
                }
            }
        }
    """
    variables = {
        "input": {
            "content": {
                "doc": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": reply_text}],
                        }
                    ],
                }
            },
            "threadId": thread_id,
        }
    }
    data = speckle_graphql_query(mutation, variables)
    return data["data"] if data and "data" in data else None


def speckle_thread(stream_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the comment thread for a given stream ID from the Speckle server.

    Args:
        stream_id (str): The ID of the stream.

    Returns:
        dict: The comment thread data is successful, None otherwise.
    """
    query = """
        query Query($projectId: String!) {
          project(id: $projectId) {
            commentThreads {
              items {
                id
                rawText
                replies {
                  items {
                    id
                    authorId
                    createdAt
                    rawText
                  }
                }
              }
            }
          }
        }
    """
    variables = {"projectId": stream_id}
    data = speckle_graphql_query(query, variables)
    return data["data"] if data and "data" in data else None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text.
    """
    pdf_reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in pdf_reader.pages)


def get_attachment(
    stream_id: str,
    attachment_id: str,
    thread_specific_filename: str,
    thread_id: str,
    folder: str = "attachments",
) -> None:
    """
    Downloads an attachment from the Speckle server and saves it to a local folder.

    Args:
        stream_id (str): The ID of the stream.
        attachment_id (str): The ID of the attachment.
        thread_specific_filename (str): The filename to save the attachment as.
        thread_id (str): The ID of the thread.
        folder (str, optional): The folder to save the attachment in. Defaults to "attachments".
    """
    url = f"{speckle_url}api/stream/{stream_id}/blob/{attachment_id}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, thread_specific_filename), "wb") as f:
            f.write(response.content)
        text = extract_text_from_pdf(os.path.join(folder, thread_specific_filename))
        nltk_text_splitter = NLTKTextSplitter(chunk_size=1000)
        chunks = nltk_text_splitter.split_text(text)
        for chunk in chunks:
            data_object = {"text": chunk, "filename": thread_specific_filename}
            weaviate_id = client.data_object.create(data_object, class_name="Documents")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def attachment(comment_id: str, thread_id: str, stream_id: str) -> Optional[str]:
    """
    Fetches the attachments for a comment from the Speckle server.

    Args:
        comment_id (str): The ID of the comment.
        thread_id (str): The ID of the thread.
        stream_id (str): The ID of the stream.

    Returns:
        str: The filename of the downloaded attachment if successful, None otherwise.
    """
    query = """
        query Query($commentId: String!, $streamId: String!) {
            comment(id: $commentId, streamId: $streamId) {
                id
                text {
                    attachments {
                        id
                        fileName
                        fileHash
                        fileType
                        fileSize
                    }
                }
            }
        }
    """
    variables = {"commentId": comment_id, "streamId": stream_id}
    data = speckle_graphql_query(query, variables)
    if data and "data" in data:
        attachments = data["data"]["comment"]["text"]["attachments"]
        for attachment in attachments:
            if attachment["fileType"].lower() == "pdf":
                thread_specific_filename = f"{thread_id}_{attachment['fileName']}"
                get_attachment(
                    stream_id, attachment["id"], thread_specific_filename, thread_id
                )
                return thread_specific_filename
            else:
                print(f"Skipped non-pdf file: {attachment['fileName']}")
    else:
        print("Failed to fetch comment attachments")
        return None


### LANGCHAIN ###

# Two options for llm, gpt3 (davinci) and gpt-3.5. The latter seems quicker and smarter but not as good at following instructions. I had more success with DaVinci. GPT-3.5 might need a custom parser to handle when it goes off track.

llm = OpenAI(temperature=0, model="text-davinci-003", openai_api_key=os.getenv["OPENAI_API_KEY"])
chat_llm = OpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv["OPENAI_API_KEY"])

# custom prompt template, changes made here greatly affect the output
template = """
You are a helpful assistant that follows instructions extremely well. Answer the question regarding a certain object in a BIM model as best you can.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:

Begin!

Question: {input}

{agent_scratchpad}
"""

# wraps the langchain summarisation feature
def gpt_standalone(text):
    prompt_template = """Your task is to analyse the provided conversation history and formulate a standalone question so that it makes sense to the receiver. Be concise; no need for pleasantries.

Conversation history:

{text}

Your forwarded question:"""

    STUFF_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=STUFF_PROMPT)

    # Converting as langchain summarise needs doc input; there may be a neater solution
    doc = [Document(page_content=text)]

    gpt_response = chain.run(doc)
    print(f"gpt_response: {gpt_response}")
    return gpt_response

# default langchain stuff from here on:


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    """
    A custom prompt template for the Langchain agent.
    """

    template: str
    tools: List[Tool]
    data_json: str = ""

    def format_messages(self, **kwargs) -> List[HumanMessage]:
        """
        Format the messages for the agent.

        Args:
            **kwargs: Keyword arguments containing the data for the messages.

        Returns:
            List[HumanMessage]: A list of formatted messages.
        """
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = "\n".join(
            f"{action.log}\nObservation: {observation}\nThought: "
            for action, observation in intermediate_steps
        )
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            f"{tool.name}: {tool.description}" for tool in self.tools
        )
        kwargs["tool_names"] = ", ".join(tool.name for tool in self.tools)
        kwargs["data_json"] = self.data_json
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


def create_prompt(tools: List[Tool]) -> BaseChatPromptTemplate:
    """
    Create a prompt for the agent.

    Args:
        tools (List[Tool]): The tools available to the agent.

    Returns:
        BaseChatPromptTemplate: The created prompt.
    """
    return CustomPromptTemplate(
        template=template, tools=tools, input_variables=["input", "intermediate_steps"]
    )


class CustomOutputParser(AgentOutputParser):
    """
    Custom Output Parser for Langchain's Agent.
    This class is used to parse the output of the language model (LLM).
    """

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse the output of the language model (LLM).

        Args:
            llm_output (str): The output from the LLM.

        Returns:
            AgentAction: If the LLM output indicates an action to be taken.
            AgentFinish: If the LLM output indicates a final answer.

        Raises:
            ValueError: If the LLM output cannot be parsed.
        """
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=llm_output)

        match = re.search(
            r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)",
            llm_output,
            re.DOTALL,
        )
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


output_parser = CustomOutputParser()

### EXECUTORS ###


# new comment:
def comment(
    data: Dict[str, Any],
    stream_id: str,
    object_id: str,
    user_question: str,
    thread_id: str,
) -> None:
    """
    Handles a comment event. It uses the data from the comment
    to initiate and run a Langchain agent, which generates a
    response. The response is then posted as a reply comment.

    Args:
        data (Dict[str, Any]): The data from the comment.
        stream_id (str): The ID of the stream the comment is in.
        object_id (str): The ID of the object the comment is about.
        user_question (str): The user's question from the comment.
        thread_id (str): The ID of the thread the comment is in.
    """

    # Insert the Speckle Data Tool from earlier here
    def get_data(input: str) -> str:
        """
        Fetches Speckle data and formats it for use by the
        Langchain agent.

        Args:
            input (str): The input string (search term).

        Returns:
            str: The formatted Speckle data.
        """
        speckle_client = get_speckle_client(speckle_host, speckle_token)

        # Get all data for the object
        speckle_data = get_object_data(speckle_client, stream_id, object_id)

        # Pretty-print JSON - Langchain and GPT will not
        # understand a Python dict
        data_formatted = json.dumps(speckle_data, indent=2)

        # Providing context with data improves GPT response
        description = (
            f"All available parameter data has been "
            f"provided below related to {input}, choose "
            f"suitable parameter value(s) matching the "
            f"question. All units are metric.\n"
        )

        description_and_data = description + data_formatted
        return description_and_data

    get_data_tool = Tool(
        name="DataSearch",
        func=get_data,
        description=(
            f"Useful when additional data is needed. All "
            f"data relevant to the question data will be "
            f"provided. After 'Action input:', you must "
            f"provide a single search string within ticks "
            f"in the following format: 'search_term'"
        ),
    )

    tools = [get_data_tool]

    comment_id = thread_id

    filename = None
    filename = attachment(comment_id, thread_id, stream_id)
    print(f"filename: {filename}")

    def weaviate_neartext(
        keyword: str, filename: str = filename, limit: int = 2
    ) -> Any:
        """
        Searches for a keyword in the attached document(s)
        using Weaviate.

        Args:
            keyword (str): The keyword to search for.
            filename (str, optional): The file name to
                                      search in. Defaults to
                                      the filename.
            limit (int, optional): The maximum number of results
                                  to return. Defaults to 2.

        Returns:
            Any: The search results.
        """
        near_text = {"concepts": keyword}
        query = (
            client.query.get("Documents", ["text", "filename"])
            .with_additional(["distance"])
            .with_near_text(near_text)
            .with_limit(limit)
        )
        if filename:
            where_filter = {
                "path": ["filename"],
                "operator": "Equal",
                "valueText": filename,
            }
            query = query.with_where(where_filter)
        results = query.do()
        return results

    weaviate_neartext_tool = Tool(
        name="DocSearch",
        func=weaviate_neartext,
        description=(
            f"Used for searching in attached document(s). "
            f"After 'Action input:', you must provide a "
            f"single search string within ticks in the "
            f"following format: 'search_term'"
        ),
    )

    if filename is not None:
        tools.append(weaviate_neartext_tool)

    tool_names = [tool.name for tool in tools]
    print(f"tool_names: {tool_names}")

    # initiate and run the langchain agent
    prompt = create_prompt(tools)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    answer = agent_executor.run(user_question)

    # Post the answer as a reply comment
    speckle_reply(thread_id, answer)


# new reply
def reply(data, stream_id, user_id, user_question, thread_id, comment_id):
    # get full comment thread, extract object_id from first comment
    raw_thread = speckle_thread(stream_id)
    conversation, object_id = extract_and_transform_thread(raw_thread, thread_id)

    print(f"conversation: {conversation}")
    print(f"object_id: {object_id}")

    # check if first comment has object_id (<25 chars)
    if len(object_id) >= 25:
        # use openai to compose standalone question from conversation history (legacy code, should probably be replaced with langchain equivalent)
        question = gpt_standalone(conversation)

        # Define a tool for Langchain agent to fetch Speckle data.
        def get_data(input: str) -> str:
            """
            Fetches Speckle data and formats it for use by the
            Langchain agent.

            Args:
                input (str): The input string (search term).

            Returns:
                str: The formatted Speckle data.
            """
            speckle_client = get_speckle_client(speckle_host, speckle_token)

            # Get all data for the object
            speckle_data = get_object_data(speckle_client, stream_id, object_id)

            # Pretty-print JSON - Langchain and GPT will not
            # understand a Python dict
            data_formatted = json.dumps(speckle_data, indent=2)

            # Providing context with data improves GPT response
            description = (
                f"All available parameter data has been "
                f"provided below related to {input}, choose "
                f"suitable parameter value(s) matching the "
                f"question. All units are metric.\n"
            )

            description_and_data = description + data_formatted
            return description_and_data

        get_data_tool = Tool(
            name="DataSearch",
            func=get_data,
            description=(
                f"Useful when additional data is needed. All "
                f"data relevant to the question data will be "
                f"provided. After 'Action input:', you must "
                f"provide a single search string within ticks "
                f"in the following format: 'search_term'"
            ),
        )

        tools = [get_data_tool]

        # Get the filenames of the current and previous attachments
        filenames = []
        attachment_filename = attachment(comment_id, thread_id, stream_id)
        if attachment_filename:
            filenames.append(attachment_filename)
        previous_attachments = [
            os.path.basename(f) for f in glob.glob(f"attachments/{thread_id}_*")
        ]
        filenames.extend(previous_attachments)

        def weaviate_neartext(keyword: str, limit: int = 2) -> dict:
            """
            Performs a near-text search in Weaviate.

            Args:
                keyword (str): The keyword to search for.
                limit (int, optional): The maximum number of results to return. Defaults to 2.

            Returns:
                dict: The search results.
            """
            near_text = {"concepts": keyword}
            query = (
                client.query.get("Documents", ["text", "filename"])
                .with_additional(["distance"])
                .with_near_text(near_text)
                .with_limit(limit)
            )
            if filenames:
                where_filter = {
                    "operator": "Or",
                    "operands": [
                        {
                            "path": ["filename"],
                            "operator": "Equal",
                            "valueString": filename,
                        }
                        for filename in filenames
                    ],
                }
                query = query.with_where(where_filter)
            return query.do()

        weaviate_neartext_tool = Tool(
            name="DocSearch",
            func=weaviate_neartext,
            description="Used for searching in attached document(s). After 'Action input:', you must provide a single search string within ticks in the following format: 'search_term'",
        )

        if filenames:
            tools.append(weaviate_neartext_tool)

        tool_names = [tool.name for tool in tools]

        # initiate and run the langchain agent
        prompt = create_prompt(tools)

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )

        answer = agent_executor.run(user_question)

        # Post the answer as a reply comment
        speckle_reply(thread_id, answer)

    return


app = Flask(__name__)
executor = Executor(app)


## ROUTE HANDLERS


@app.route("/comment-webhook", methods=["POST"])
def handle_webhook():
    """
    Handles incoming webhook requests.

    Returns:
        str: An empty string.
        int: HTTP status code 200.
    """
    data = request.get_json()
    event_name = data["payload"]["event"]["event_name"]
    print(f"event_name: {event_name}")

    if event_name == "comment_created":
        print("new comment")
        stream_id, comment_text, thread_id = extract_comment_data(data["payload"])

        # Check if the comment begins with @Spackle
        # and has a valid object ID
        if comment_text.startswith("@Spackle"):
            print("ignored: invalid prompt")
        else:
            print("valid prompt detected")

            object_id = re.search(r"\b\w{20,}\b", comment_text)
            # If a matching string is found, use it as the object_id
            # If not, return None
            object_id = object_id.group() if object_id else None

            # If no object ID is provided, reply asking for it
            if len(object_id) <= 20:
                print("Object ID missing")
                bot_reply = "Hi, I'm Spackle. To assist you, I need a valid object ID."

                ### We will add a reply function later

            else:
                executor.submit(
                    reply, data, stream_id, user_id, comment_text, thread_id, comment_id
                )

    elif event_name == "comment_replied":
        print("new reply")
        stream_id, user_id, reply_text, thread_id, comment_id = extract_reply_data(
            data["payload"]
        )

        # Check if the bot did not generate the reply
        # and is part of a registered prompt conversation
        if reply_text.startswith("@Spackle"):
            print("ignored: reply from bot or unregistered conversation")
        else:
            print("reply to registered conversation detected")

            executor.submit(
                reply, data, stream_id, user_id, reply_text, thread_id, comment_id
            )

    return "", 200


if __name__ == "__main__":
    if not os.path.exists("webhook"):
        os.makedirs("webhook")
    app.run(host=flask_url, port=flask_port)
