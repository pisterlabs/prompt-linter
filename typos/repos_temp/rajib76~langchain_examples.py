f"""
You are a helpful chatbot. 
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be based on your knowledge
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""f"""
                version: 2

                sources:
                    - name: source_01
                      description: This is a replica of the Snowflake database used by our app
                      database: pc_dbt_db
                      schema: dbt_rdeb
                      tables:
                          - name: customer
                            description: This the final customer table.
                          - name: stg_customer
                            description: the customer table for staging.
                          - name: stg_orders
                            description: One record per order. Includes cancelled and deleted orders."""f"""
        version: 2

        models:
          - name: customer
            description: One record per customer
            columns:
              - name: customer_id
                description: Primary key
                tests:
                  - unique
                  - not_null
              - name: first_name
                description: The first name of the customer
              - name: last_name
                description: The last name of the customer
              - name: first_order_date
                description: NULL when a customer has not yet placed an order.
              - name: most_recent_order_date
                description: customers most recent date of order.
              - name: number_of_orders
                description: total number of orders by the customer
        
          - name: stg_customers
            description: This model cleans up customer data
            columns:
              - name: customer_id
                description: Primary key to identify a customer
                tests:
                  - unique
                  - not_null
              - name: first_name
                description: First name of the customer
              - name: last_name
                description: last name of the customer                
        
          - name: stg_orders
            description: This model cleans up order data
            columns:
              - name: order_id
                description: Primary key
                tests:
                  - unique
                  - not_null
              - name: customer_id
                description: Primary key to identify a customer
              - name: order_date
                description: date when customer placed the order.                
              - name: status
                tests:
                  - accepted_values:
                      values: ['placed', 'shipped', 'completed', 'return_pending', 'returned']""""""You are a helpful reviewer. You review business requirements against functional requirements.
        You will be given a business requirement which you will need to match with the functional requirement provided in the context.
        Answer the question based only on the context provided. Do not make up your answer.
        Answer in the desired format given below.

        Desired format:
        Business requirement: The business requirement given to compare against functional requirement
        Functional requirement: The content of the functional requirement

        {context}
        {question}
        """"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
{context}
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""""""
Context:{context}
User: {query}
AI: {answer}
""""""The following are exerpts from comversation with an AI assistant
who understands life events. Please ensure that you are correctly classifying a life event.
Life events are a change of a situation in someone's life and only the below scenarios are applicable
to consider the event as a life event

    - Losing existing health coverage, including job-based, individual, and student plans
    - Losing eligibility for Medicare, Medicaid, or CHIP
    - Turning 26 and losing coverage through a parent’s plan
    - Getting married or divorced
    - Having a baby or adopting a child
    - Death in the family
    - Moving to a different ZIP code or county
    - A student moving to or from the place they attend school
    - A seasonal worker moving to or from the place they both live and work
    - Moving to or from a shelter or other transitional housing
    - Changes in your income that affect the coverage you qualify for
    - Gaining membership in a federally recognized tribe or status as an Alaska Native Claims Settlement Act (ANCSA) Corporation shareholder
    - Becoming a U.S. citizen
    - Leaving incarceration (jail or prison)
    - AmeriCorps members starting or ending their service

Here are the examples
""""""
Context:{context}
User:{query}
AI: 
""""""
Context: {context}
User: {query}
AI: {answer}
""""""
Context: {context}
User: {query}
AI:
""""""You are a helpful cobol programmer. You will understand the logic of cobol programs 
and help identify enhancements that are required withing the program and the subprograms
based on the code snippet provided as context. 
Answer the question based only on the context provided. Do not make up your answer.
Answer in the desired format given below.

Desired format:
Program Name: The name of the program which requires change
Code snippet: The piece of code that requires a change

{context}
{question}
""""""Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return {no_output_str}.

Remember, *DO NOT* edit the extracted parts of the context.

> Question: {{question}}
> Context:
>>>
{{context}}
>>>
Extracted relevant parts:""""""Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Craft the final answer to the original input question based on tool output""""""Begin!

Question: {input}
Thought:{agent_scratchpad}""""""
Context: {context}
User: {query}
AI: {answer}
""""""
Context: {context}
User: {query}
AI:
""""""
You are assessing a submitted answer on a given question based on a provided context and criterion. Here is the data: 
[BEGIN DATA] 
*** [answer]:    {answer} 
*** [question]:  {question} 
*** [context]:   {context}
*** [Criterion]: {criteria} 
*** 
[END DATA] 
Does the submission meet the criterion? First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print "Correct" or "Incorrect" (without quotes or punctuation) on its own line corresponding to the correct answer.
The answer will be correct only if it meets all the criterion.
Reasoning:""""""Question: {question}

Answer: Let's think step by step.""""""Use provided tool to moderate the response:

{response}""""""Answer the below question:

{question}"""