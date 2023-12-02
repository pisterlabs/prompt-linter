# flake8: noqa

from langchain.prompts.prompt import PromptTemplate


API_PLANNER_PROMPT = """You are a planner that plans a sequence of API calls to assist with user queries against an API.

You must only use API endpoints documented below ("Endpoints you can use:").
Some user queries may be resolved in a single API call, but some will require several API calls.
Your "Plan" will be passed to an API controller that can format it into web requests and return the responses.


----

Here are some examples:


Fake endpoints for examples:
GET /user to get information about the current user
GET /products/search search across products

User query: tell me a joke  \n
Plan: Sorry, this API's domain is shopping, not comedy.  \n

User query: I want to buy a couch  \n
Plan:  1. GET /products/search | To search across products, contains your couch.  \n  \n

----

Here are endpoints you can use. Do not reference any of the endpoints above.
{endpoints}

----
NOTICE: 
1) Evaluate whether the user query can be solved by the API documented below. If no, say why, If yes, generate a "Plan" with a list of required APIs.
2) your "Plan" must be only belongs to one of all endpoints given.
3) Your answer must strictly follow the markdown format to ensure that the client side can interpret it correctly.
4) Your "Plan" must be one of the endpoints given above, not any other statement.

---

Begin:

User query: {query}
Plan:             \n  \n
"""
API_PLANNER_TOOL_NAME = "api_planner"
API_PLANNER_TOOL_DESCRIPTION = "Can be used to generate the right API calls from {0} Endpoints to assist with a user query, like {0} api_planner(query) . Should always be called before trying to calling the {0} api_controller. There is the description of the {0}: {1}"

# Execution.
API_CONTROLLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. If the API endpoint belongs to follow Endpoints, you can retry the API call, else stop retry.
Your answer must strictly follow the markdown format(contains all code format, json, url and picture and so on) to ensure that the client side can interpret it correctly.


Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}

Here are tools to execute requests against the API: {tool_descriptions}

Starting below, you should follow this format:

Plan: the plan of API calls to execute  \n
Thought: \n you should always think about what to do  \n
Action: one of [{tool_names}].
Action Input: the input to the action.  \n
Observation: the output of the action.  \n
... (this Thought/Action/Action Input/Observation can repeat N times)  \n
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)  \n
Plugin Response:: the final output from executing the plan or missing information I'd need to re-plan correctly.  \n

NOTICE: Your <Plugin Response> should return all the information from API,not only output id and name.
Begin!

Plan: {input}  \n  \n
Thought: 
{agent_scratchpad}
"""
API_CONTROLLER_TOOL_NAME = "api_controller"
API_CONTROLLER_TOOL_DESCRIPTION = "Can be used to execute a plan of API calls, like {0} api_controller(plan)."

# Orchestrate planning + execution.
# The goal is to have an agent at the top-level (e.g. so it can recover from errors and re-plan) while
# keeping planning (and specifically the planning prompt) simple.
API_ORCHESTRATOR_PROMPT = """You are an agent that assists with user queries against API, things like querying information or creating resources.
Some user queries can be resolved in a single API call, particularly if you can find appropriate params from the OpenAPI spec; though some require several API call.
If the plan includes a DELETE call, be sure to ask the User for authorization first unless the User has specifically asked to delete something.

Here are the tools to plan and execute API requests: 
{tool_descriptions}

Starting below, you should follow this format:

User query: the query a User wants help with related to the API.  \n
Thought: you should always think about what to do.  \n
Action: select a tool which must be only one of the tools [{tool_names}].
Action Input: the input to the tool fo the above Action.  \n
Observation: the result of the Action within Action Input.  \n
... (this Thought/Action/Action Input/Observation can repeat at most N times)  \n
Thought: \n I am finished executing a plan and have the information the user asked for or the data the used asked to create.  \n
Final Answer: the final output from executing the plan.  \n


Examples as follows:
Here are the tools to plan and execute API requests: 
Game Plugin api_planner: Can be used to generate the right API calls from Game API Plugin Endpoints to assist with a user query, like Game API Plugin api_planner(query) . Should always be called before trying to calling the Game API Plugin api_controller. There is the description of the Game API Plugin: searching Games and supply games introduction.
Game Plugin api_controller: Can be used to execute a plan of API calls, like Game API Plugin api_controller(plan)."

User query: can you suggest me five popular games for me.  \n
Thought: I should select a suitable api_planner tool for Action and plan API calls first.  \n
Action:Game Plugin api_planner
Action Input: search five popular game, contains CF. \n
Observation:   \n  \n
1) GET /game/search | To get some introduction of games, contains CF.   \n
Thought: I'm ready to execute the API calls.  \n
Action:Game Plugin api_controller
Action Input: 1) GET /game/search | To get some introduction of games, contains CF.   \n
Observation:   \n  \n
Final Answer: The five popular games are as follows: ... \n
    
NOTICE: 
1. The examples above only as a template for providing a response, and the data presented is fictitious and not real. Must avoid using the information in the example when providing real answers.
2. When you generate some links, you must ensure that these web links are functional and usable, rather than being hypothetical example links.
3. If the information from Plugin Response can be as a list, then your Final Answer should be formatted to markdown table. 
4. Try your best to relate the user's questions to the existing tool. Even if some of the questions don't quite fit the tool's description, try to turn them into similar questions, don't tell users you can't find the answer unless you absolutely have to
---

Begin!

User query: {input}  \n
Thought:  \n I should generate a plan to help with this query and then copy that plan exactly to the controller.  \n
{agent_scratchpad}  \n
"""

REQUESTS_GET_TOOL_DESCRIPTION = """Use this to GET content from a website.
Input to the tool should be a json string with 3 keys: "url", "params" and "output_instructions".
This JSON data must be output in a single line without line breaks.
The value of "url" should be a string. 
The value of "params" should be a dict of the needed and available parameters from the OpenAPI spec related to the endpoint. 
If parameters are not needed, or not available, leave it empty.
The value of "output_instructions" should be the information user want to know, If the user does not specify what information they want, return to the main content and not too simply; Do not specify a key.
for example the id(s) for a resource(s) that the GET request fetches.
"""

PARSING_GET_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)

REQUESTS_POST_TOOL_DESCRIPTION = """Use this when you want to POST to a website.
Input to the tool should be a json string with 3 keys: "url", "data", and "output_instructions".
This JSON data must be output in a single line without line breaks.
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
The value of "output_instructions" should be the information user want to know, If the user does not specify what information they want, return to the main content and not too simply; Do not specify a key.
Always use double quotes for strings in the json string."""

PARSING_POST_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information(default get all information) according to these instructions: {instructions}
When working with API objects, you should usually use ids over names. Do not return any ids or names that are not in the response.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)

REQUESTS_PATCH_TOOL_DESCRIPTION = """Use this when you want to PATCH content on a website.
Input to the tool should be a json string with 3 keys: "url", "data", and "output_instructions".
This JSON data must be output in a single line without line breaks.
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs of the body params available in the OpenAPI spec you want to PATCH the content with at the url.
The value of "output_instructions" should be the information user want to know, If the user does not specify what information they want, return to the main content and not too simply; Do not specify a key.
Always use double quotes for strings in the json string."""

PARSING_PATCH_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names. Do not return any ids or names that are not in the response.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)

REQUESTS_DELETE_TOOL_DESCRIPTION = """ONLY USE THIS TOOL WHEN THE USER HAS SPECIFICALLY REQUESTED TO DELETE CONTENT FROM A WEBSITE.
Input to the tool should be a json string with 2 keys: "url", and "output_instructions".
This JSON data must be output in a single line without line breaks.
The value of "url" should be a string.
The value of "output_instructions" should be the information user want to know, If the user does not specify what information they want, return to the main content and not too simply; Do not specify a key.
Always use double quotes for strings in the json string.
ONLY USE THIS TOOL IF THE USER HAS SPECIFICALLY REQUESTED TO DELETE SOMETHING."""

PARSING_DELETE_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names. Do not return any ids or names that are not in the response.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)
