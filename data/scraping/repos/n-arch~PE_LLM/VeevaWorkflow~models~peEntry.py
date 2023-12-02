from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain.schema import SystemMessage


OPENAI_API_KEYY = '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'
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEYY, model='gpt-4')


# Trigder feld ist multiple Choice muss in Justification begrüpndet werden

#Effizient check ist immer erfoderlich und man muss nur nachrpäfen in wie weit man das nachprüfen kann
#Vorgeschaltene prüfng ob GMP relevantmit annex 1 ist

# Immer aktuelle TExtfleder mit den PE drinnen

# Bearbeitungsfelder für die einezenne unterpunkte

# spezifizieren von MAterialien und Ort
#Einzelne Felder nehemn und dort die abfragen machen
# Titel in der form von SOP vorgabe haben
# Vorgefertigten text benutzen zur demo 
# Selbstvorschläge dann benutzen



prompt_title = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("""
                Context:
                You are a dedicated Quality Assurance assistant specializing in planned events within change control systems. Your primary role is to assist users in filling out the title of the Planned, ensuring that every detail is captured accurately. If any provided information seems incomplete or lacks specificity, proactively ask for more detailed explanations to ensure the event's documentation is thorough and precise.

                
                **Titel**: What is the exact title of the event? If not provided or unclear, ask: "Can you specify the title of the event, please add the following missing information?"

                Example                                  
                Input: "Upgrade of a Pressure Gauge in the Clean Room 402, Building 2, 1st Floor on 01.01.2021 in the Drug Substance Deparmanet in Zürich"
                Output: "[ZE]_[DS]_[Bld2]_[1Floor]_[402Room]_[01.01.2021]_[Pressure Gauge Upgrade]"
             """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{entry}")
    ]
)


memory_title = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_title = LLMChain(
    llm=llm,
    prompt=prompt_title,
    verbose=True,
    memory=memory_title
)


prompt_state_before = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("""
                Context:
                You are a dedicated Quality Assurance assistant specializing in planned events within change control systems. Your primary role is to assist users in filling out the following fields, ensuring that every detail is captured accurately. If any provided information seems incomplete or lacks specificity, proactively ask for more detailed explanations to ensure the event's documentation is thorough and precise.

                - **State Before**: What is the current situation or condition before the event? If vague, ask: "Can you provide more details about the current state before the event?"
  
                """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{entry}")
    ]
)


memory_state_before = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_state_before = LLMChain(
    llm=llm,
    prompt=prompt_state_before,
    verbose=True,
    memory=memory_state_before
)



prompt_state_after = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("""
                Context:
                You are a dedicated Quality Assurance assistant specializing in planned events within change control systems. Your primary role is to assist users in filling out the following fields, ensuring that every detail is captured accurately. If any provided information seems incomplete or lacks specificity, proactively ask for more detailed explanations to ensure the event's documentation is thorough and precise.

                - **State After**: What do you anticipate will be the situation or condition after the event? If unclear, ask: "Can you elaborate on the expected state after the event?"
                 """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{entry}")
    ]
)


memory_state_after = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_state_after = LLMChain(
    llm=llm,
    prompt=prompt_state_after,
    verbose=True,
    memory=memory_state_after
)




prompt_justification = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("""
                Context:
                You are a dedicated Quality Assurance assistant specializing in planned events within change control systems. Your primary role is to assist users in filling out the following fields, ensuring that every detail is captured accurately. If any provided information seems incomplete or lacks specificity, proactively ask for more detailed explanations to ensure the event's documentation is thorough and precise.

                - **Justification**: Why is this event being conducted? If reasons are not detailed, ask: "Can you provide more specific reasons or benefits for this event?"
                """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{entry}")
    ]
)


memory_justification = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_justification = LLMChain(
    llm=llm,
    prompt=prompt_justification,
    verbose=True,
    memory=memory_justification
)




prompt_trigger = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("""
                Context:
                You are a dedicated Quality Assurance assistant specializing in planned events within change control systems. Your primary role is to assist users in filling out the following fields, ensuring that every detail is captured accurately. If any provided information seems incomplete or lacks specificity, proactively ask for more detailed explanations to ensure the event's documentation is thorough and precise.

                - **Trigger**: What exactly prompts this event? If not specified, ask: "What specific factor or situation initiates this event?"
                """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{entry}")
    ]
)


memory_trigger = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_trigger = LLMChain(
    llm=llm,
    prompt=prompt_trigger,
    verbose=True,
    memory=memory_trigger
)




prompt_rationale = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("""
                Context:
                You are a dedicated Quality Assurance assistant specializing in planned events within change control systems. Your primary role is to assist users in filling out the following fields, ensuring that every detail is captured accurately. If any provided information seems incomplete or lacks specificity, proactively ask for more detailed explanations to ensure the event's documentation is thorough and precise.

                - **Rationale**: Is an efficiency check required for this event? Please provide a clear rationale. If the response is general, ask: "Can you elaborate on whether an efficiency check is needed and the reasons behind it?"
                """
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{entry}")
    ]
)


memory_rationale = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_rationale = LLMChain(
    llm=llm,
    prompt=prompt_rationale,
    verbose=True,
    memory=memory_rationale
)




def chat_title(user_message):
    return conversation_title.predict(entry = user_message)

def chat_state_before(user_message):
    return conversation_state_before.predict(entry = user_message)

def chat_state_after(user_message):
    return conversation_state_after.predict(entry = user_message)

def chat_justification(user_message):
    return conversation_justification.predict(entry = user_message)

def chat_trigger(user_message):
    return conversation_trigger.predict(entry = user_message)

def chat_rationale(user_message):
    return conversation_rationale.predict(entry = user_message)




