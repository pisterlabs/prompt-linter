import openai
try:
    import app.config as config
    from app.slot_fillings import extract_slots
    from app.load_intent_model import predictModel
except:
    import config as config
    from slot_fillings import extract_slots
    from load_intent_model import predictModel


openai.api_key = config.open_ai_api
model_engine = config.model_engine
chatbot_prompt = config.chatbot_prompt




def get_response(user_input="",chat_log=None, model=None):
    """generate bot response with intent and slot fillings"""
    
    if chat_log==None:
        chat_log=chatbot_prompt
        
    
    prompt = chatbot_prompt.replace(
        "<conversation_history>", chat_log).replace("<user input>", user_input)

    # Get the response from GPT-3
    response = openai.Completion.create(
        engine=model_engine, prompt=chat_log, max_tokens=2048, n=1, stop=None, temperature=0.5)

    # Extract the response from the response object
    response_text = response["choices"][0]["text"]

    chatbot_response = response_text.strip()
    try:
        intent_result = model.predict(str(user_input))
    except:
        intent_result= "Intent classifier Model doesn't exist"
        print(intent_result)
    slot_fillings = extract_slots(user_input)
    return {"Chatbot": chatbot_response, "intent":intent_result, "slot_filling":slot_fillings}


def append_to_chat_log(question,answer,chat_log):
    """store session or chat logs"""
    print(f'{chat_log}')
    return f'Chatbot: {answer}\n user: {question}\n'

