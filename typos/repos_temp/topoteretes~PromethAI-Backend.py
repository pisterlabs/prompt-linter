""" Hey ChatGPT, I need your help in decomposing the following task into a series of manageable steps for the purpose of task identification based on 
                    Newell and Simon paper. Return the result as a json with the result type 'Identification' and 'Value': 'Decomposition'  : {task_description}"""""" Hey ChatGPT, I need your help in creating an analogy for the purpose of task identification based on 
                    Newell and Simon paper. Return the result as a json with the result type 'Identification' and 'Value': 'Analogy'  : {task_description}""""""

            Based on all the history and information of this user, decide based on user query query: {query} which of the following tasks needs to be done:
            1. Memory retrieval , 2. Memory update,  3. Convert data to structured   If the query is not any of these, then classify it as 'Other'
            Return the result in format:  'Result_type': 'Goal', "Original_query": "Original query"
            """""" The {name} has following {past_preference} and the new {preferences}
                Update user preferences and return a list of preferences
            Do not embellish.
            Summary: """""" The {name} has following {past_dislikes} and the new {dislikes}
                Update user taboos and return a list of dislikes
            Do not embellish.
            Summary: """""" The {name} has following {past_traits} and the new {traits}
                Update user traits and return a list of traits
            Do not embellish.
            Summary: """""" Create a food recipe based on the following prompt: '{{prompt}}'. Instructions and ingredients should have medium detail.
                Answer a condensed valid JSON in this format: {{ json_example}}  Do not explain or write anything else."""""" Decompose decision point '{{ base_category }}' into three categories the same level as value '{{base_value}}'  definitely including '{{base_value}} ' but not including  {{exclusion_categories}}. Make sure choices further specify the  '{{ base_category }}' category  where AI is helping person in choosing {{ assistant_category }}.
        Provide three sub-options that further specify the particular category better. Generate very short json, do not write anything besides json, follow this json property structure : {{json_example}}""""""
               Decompose {{ prompt_str }} statement into decision tree that take into account user summary information and related to {{ assistant_category }}. There should be three categories and one decision for each.  
               Categories should be logical and user friendly. Do not include budget, meal type, intake, personality, user summary, personal preferences.
               Decision should be one user can make in regards to {{ assistant_category }}. Present answer in one line and in property structure : {{json_example}}""""""Change the category: {{category}} based on {{from_}} to {{to_}}  change and update appropriate of the following original inluding the preference: {{results}}
         """"""
              Based on the following prompt {{prompt}} and all the history and information of this user,
                Determine the type of restaurant you should offer to a customer. Make the recomendation very short and to a point, as if it is something you would type on google maps
            """"""
              Based on the following prompt {{prompt}}
                Determine the type of food you would want to recommend to the user, that is commonly ordered online. It should of type of food offered on a delivery app similar to burger or pizza, but it doesn't have to be that.
                The response should be very short
            """"""

            Based on all the history and information of this user, classify the following query: {query} into one of the following categories:
            1. Goal update , 2. Preference change,  3. Result change 4. Subgoal update  If the query is not any of these, then classify it as 'Other'
            Return the classification and a very short summary of the query as a python dictionary. Update or replace or remove the original factors with the new factors if it is specified.
            with following python dictionary format 'Result_type': 'Goal', "Result_action": "Goal changed", "value": "Diet added", "summary": "The user is updating their goal to lose weight"
            Make sure to include the factors in the summary if they are provided
            """