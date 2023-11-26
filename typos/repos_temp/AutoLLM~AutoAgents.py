"""We are working together to satisfy the user's original goal
step-by-step. Play to your strengths as an LLM. Make sure the plan is
achievable using the available tools. The final answer should be descriptive,
and should include all relevant details.

Today is {today}.

## Goal:
{input}

If you require assistance or additional information, you should use *only* one
of the following tools: {tools}.

## History
{agent_scratchpad}

Do not repeat any past actions in History, because you will not get additional
information. If the last action is Tool_Search, then you should use Tool_Notepad to keep
critical information. If you have gathered all information in your plannings
to satisfy the user's original goal, then respond immediately with the Finish
Action.

## Output format
You MUST produce JSON output with below keys:
"thought": "current train of thought",
"reasoning": "reasoning",
"plan": [
"short bulleted",
"list that conveys",
"next-step plan",
],
"action": "the action to take",
"action_input": "the input to the Action",
""""""We are working together to satisfy the user's original goal
step-by-step. Play to your strengths as an LLM. Make sure the plan is
achievable using the available tools. The final answer should be descriptive,
and should include all relevant details.

Today is {today}.

## Goal:
{input}

If you require assistance or additional information, you should use *only* one
of the following tools: {tools}.

## History
{agent_scratchpad}

Do not repeat any past actions in History, because you will not get additional
information. If the last action is Tool_Wikipedia, then you should use Tool_Notepad to keep
critical information. If you have gathered all information in your plannings
to satisfy the user's original goal, then respond immediately with the Finish
Action.

## Output format
You MUST produce JSON output with below keys:
"thought": "current train of thought",
"reasoning": "reasoning",
"plan": [
"short bulleted",
"list that conveys",
"next-step plan",
],
"action": "the action to take",
"action_input": "the input to the Action",
"""""" Useful for when you need to ask with search. Use direct language and be
EXPLICIT in what you want to search. Do NOT use filler words.

## Examples of incorrect use
{
     "action": "Tool_Search",
     "action_input": "[name of bagel shop] menu"
}

The action_input cannot be None or empty.
"""""" Useful for when you need to note-down specific
information for later reference. Please provide the website and full
information you want to note-down in the action_input and all future prompts
will remember it. This is the mandatory tool after using the Tool_Search.
Using Tool_Notepad does not always lead to a final answer.

## Examples of using Notepad tool
{
    "action": "Tool_Notepad",
    "action_input": "(www.website.com) the information you want to note-down"
}
"""""" Useful for when you need to note-down specific
information for later reference. Please provide the website and full
information you want to note-down in the action_input and all future prompts
will remember it. This is the mandatory tool after using the Tool_Wikipedia.
Using Tool_Notepad does not always lead to a final answer.

## Examples of using Notepad tool
{
    "action": "Tool_Notepad",
    "action_input": "(www.website.com) the information you want to note-down"
}
"""""" Useful for when you need to get some information about a certain entity. Use direct language and be
concise about what you want to retrieve. Note: the action input MUST be a wikipedia entity instead of a long sentence.
                        
## Examples of correct use
1.  Action: Tool_Wikipedia
    Action Input: Colorado orogeny

The Action Input cannot be None or empty.
"""""" This tool is helpful when you want to retrieve sentences containing a specific text snippet after checking a Wikipedia entity. 
It should be utilized when a successful Wikipedia search does not provide sufficient information. 
Keep your lookup concise, using no more than three words.

## Examples of correct use
1.  Action: Tool_Lookup
    Action Input: eastern sector

The Action Input cannot be None or empty.
"""""" Useful when you have enough information to produce a
final answer that achieves the original Goal.

You must also include this key in the output for the Tool_Finish action
"citations": ["www.example.com/a/list/of/websites: what facts you got from the website",
"www.example.com/used/to/produce/the/action/and/action/input: "what facts you got from the website",
"www.webiste.com/include/the/citations/from/the/previous/steps/as/well: "what facts you got from the website",
"www.website.com": "this section is only needed for the final answer"]

## Examples of using Finish tool
{
    "action": "Tool_Finish",
    "action_input": "final answer",
    "citations": ["www.example.com: what facts you got from the website"]
}
""""""We are using the Search tool.
                 # Previous queries:
                 {history_string}. \n\n Rewrite query {action_input} to be
                 different from the previous queries."""