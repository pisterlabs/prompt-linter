import inspect
from chatbot_service import ChatBotService
from langchain_service import LangChainService
from knowledge_base_service import KnowledgeBaseService
from env_setter import setup_keys
import os


# Define an external function to inspect an object
def inspect_object(obj, parent_class_name=None):
    # Inspect the passed object
    class_name = obj.__class__.__name__
    print("Name of the passed object's class:", class_name)
    
    # Indicate if the object was passed into the constructor of another class
    if parent_class_name:
        print(f"The object '{class_name}' is passed into the constructor of '{parent_class_name}'")
    
    # Get the methods of the passed object (excluding dunder methods)
    methods = [attr for attr in dir(obj) 
               if callable(getattr(obj, attr)) and not attr.startswith('__') and not attr.startswith('_')]
    # Convert the list of methods to a string and print without brackets
    print(f"Methods of the passed object '{class_name}' (excluding dunder methods and private): {', '.join(methods)}")
    
    # Recursively inspect the object passed into the constructor (if it exists)
    for attr_name, attr_value in obj.__dict__.items():
        if isinstance(attr_value, (LangChainService, KnowledgeBaseService)):
            passed_obj = attr_value
            passed_obj_name = passed_obj.__class__.__name__
            print(f"\nThe class '{class_name}' has an object of class '{passed_obj_name}' passed into its constructor")
            inspect_object(passed_obj, parent_class_name=class_name)





chatOpenAI : ChatOpenAI = ChatOpenAI(
            temperature=0, openai_api_key=self.chatbotSettings.OPENAI_API_KEY)
chatbotSettings = ChatBotSettings()
langchain_service = LangChainService(chatbotSettings,chatOpenAI)
knowledge_base_service = KnowledgeBaseService()


# Create an instance of OuterClass and pass an object to its constructor
outer_obj = ChatBotService(langchain_service, knowledge_base_service)

# Call the external function to inspect the object from outside the class
inspect_object(outer_obj)
