from google.cloud import translate_v2 as translate
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from utils import VideoMapUtil, insert_message, get_chat_history

# Create your own message templates:

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from utils import insert_message, get_chat_history

# Create your own message templates:

# system_template = SystemMessagePromptTemplate.from_template(
#     "This is Farmer CHAT, a friendly and knowledgeable AI here to help the farming community with only farming-related queries."
#     "You can access a comprehensive library of farming resources, including various PDF documents."
#     "Your role is to assist users by answering their queries about only farming  using the information available in these resources. Your responses should be detailed and accurate."
#     "Make sure you are friendly, conversational and always ask follow-up questions."
#     "Format all your answers using bullet points, new lines to increase readability."
#     "Decorate the answer with relevant emojis compatible with Telegram."
#     "If user asks a question related to anything other than farming say that you donot know."
# )
system_template = SystemMessagePromptTemplate.from_template(
    "This is Farmer CHAT, a highly knowledgeable assistant specializing in chilli farming, here to help the farming community."
    "You can access only the embeddings which are sent."
    "Your role is to assist users by answering their queries about chilli farming using the information available in these resources. Your responses should be detailed and accurate."
    "Make sure you are friendly, conversational and always ask follow-up questions. "
    "Format all your answers using bullet points, new lines to increase readability. "
    "Decorate the answer with relevant emojis compatible with Telegram."
    "If you are unable to find the answer from the provided embeddings, just respond I am sorry, this question is out of my scope."
)

# input_suffix = "Also ask any follow up questions if you find it useful."

human_template = HumanMessagePromptTemplate.from_template("{text}")

# Then, build a ChatPromptTemplate from your message templates:
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_template, human_template]
)


# def chat_prompt_with_history(chat_prompt_template, history, input):
#     history_template = "Relevant parts of our earlier conversation:\n{history}\n(Note: These details are available for reference and can be used if they're relevant to the current question.)"
#     formatted_history = history_template.format(history=get_chat_history())
#     current_conversation = (
#         "\n\nCurrent conversation:\nFarmer: {input}\nFarmer CHAT:".format(input=input)
#     )

#     return f"{chat_prompt_template}\n\n{formatted_history}{current_conversation}"


def main_bot_logic(user_id, user_input, qa, video_vectordb):
    translate_client = translate.Client()

    try:
        result = translate_client.detect_language(user_input)
        input_language = result["language"]
        print("*****", input_language)

        if input_language in ["hi"]:
            print("Its hindi bro..")
            translation = translate_client.translate(user_input, target_language="en")
            chat_prompt = chat_prompt_template.format_prompt(
                input_language="hi",
                output_language="en",
                text=translation["translatedText"],
            ).to_string()
            translated_input_message = translation["translatedText"]
        else:
            translated_input_message = user_input
            chat_prompt = chat_prompt_template.format_prompt(
                input_language=input_language,
                output_language="English",
                text=user_input,
            ).to_string()

        docs = video_vectordb.similarity_search_with_score(translated_input_message)

        video_url = None
        if len(docs) > 0:
            if docs[0][1] < 0.3:
                video_source = docs[0][0].metadata.get("source")
                if video_source:
                    video_url = VideoMapUtil.video_map.get(video_source).get("hindi")

        llm_response = qa(
            {"question": chat_prompt, "chat_history": get_chat_history(user_id)}
        )

        if llm_response:
            response_text = llm_response["answer"]
            message_translated_response = response_text

            if video_url:
                response_text = response_text + "\n Here is a video which I found : "

            if input_language in ["hi"]:
                translation = translate_client.translate(
                    response_text, target_language=input_language
                )
                message_translated_response = translation["translatedText"]
                response = translation["translatedText"]
            else:
                response = response_text

            message_translated_response = (
                (message_translated_response + video_url)
                if video_url
                else message_translated_response
            )
            response_text = (response_text + video_url) if video_url else response_text
            response = (response + video_url) if video_url else response

            print("Here", message_translated_response)
            message = insert_message(
                user_id,
                user_input,
                response_text,
                translated_input_message,
                message_translated_response,
            )
            return response, message
        else:
            if input_language == "hi":
                return (
                    "क्षमा करें, मैं आपकी सहायता करने में असमर्थ था। कृपया फिर से प्रयास करें।",
                    None,
                )
            else:
                return "Sorry, I was unable to assist you. Please try again.", None
    except Exception as err:
        print("Exception occurred. Please try again", str(err))
        if input_language == "hi":
            return "क्षमा करें, कुछ गलत हुआ। कृपया फिर से प्रयास करें।", None
        else:
            return "Sorry, something went wrong. Please try again.", None
