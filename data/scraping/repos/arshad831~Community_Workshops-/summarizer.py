import os
import openai
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

# openai key from the .env file

api_key , org_id = sk.openai_settings_from_dot_env()

kernel.add_text_completion_service(
    "dv",OpenAIChatCompletion("gpt-4", api_key,org_id ))

#print(prompt())
#prompt template
summarize = kernel.create_semantic_function(
    "{{$input}}\n\nGive me a summary in exactly 5 words"
)

print(summarize(""" 
                1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.
                
                """))


print(summarize(""" 
1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
2. The acceleration of an object depends on the mass of the object and the amount of force applied.
3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first
                
                """))
