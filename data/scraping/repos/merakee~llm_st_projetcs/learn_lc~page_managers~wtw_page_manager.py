# pyhton
from enum import Enum

# local
from st_code.st_session_manager import STSessionManager
from app_managers.llm_manager import LLMManager

# langchain
from langchain.prompts import PromptTemplate


class WTWSessionKeys(Enum):
    whattowatch_type = 1
    whattowatch_genre = 2
    whattowatch_count = 3


class WTWSessionManager():
    @staticmethod
    def get_value_for_key(sesssion_key):
        return STSessionManager.get_value_for_key(session_key=sesssion_key)

# implementation


class WTWPageManager:
    @staticmethod
    def get_template():
        template = "I want to watch some entertaining and popular shows. I am in mood for some {genre} shows. Please suggtest me {count} {type}."
        return template

    def get_prompt():
        prompt = PromptTemplate(
            input_variables=["count", "type", "genre"],
            template=WTWPageManager.get_template())
        wtw_count = WTWSessionManager.get_value_for_key(
            WTWSessionKeys.whattowatch_count)
        wtw_type = WTWSessionManager.get_value_for_key(
            WTWSessionKeys.whattowatch_type)
        wtw_genre = WTWSessionManager.get_value_for_key(
            WTWSessionKeys.whattowatch_genre)

        prompt_complete = prompt.format(
            count=wtw_count, type=wtw_type, genre=wtw_genre)
        return prompt_complete

    def get_type_list():
        return ['TV Show', 'Movie',
                'TV Show or Movie']

    def get_genre_list():
        return ('Drama', 'Comedy', 'Action', 'Documentary', 'Reality TV', 'Sci-fi', 'Crime', 'Animation', 'Game Show')

    def get_whattowatch_count():
        count = WTWSessionManager.get_value_for_key(
            WTWSessionKeys.whattowatch_count)
        if not count:
            count = 3
            STSessionManager.set_key(
                WTWSessionKeys.whattowatch_count, count)

        return count

    def get_index_for_key(session_key):
        if session_key.name == WTWSessionKeys.whattowatch_type:
            return WTWPageManager.get_type_list().index(WTWSessionManager.get_value_for_key(session_key))
        if session_key.name == WTWSessionKeys.whattowatch_genre:
            return WTWPageManager.get_genre_list().index(WTWSessionManager.get_value_for_key(session_key))

        return 0

    def get_llm_response():
        prompt = WTWPageManager.get_prompt()
        # check if api_keyis set
        if not STSessionManager.is_api_key_set():
            response = "API Key not set. LLM not run"
            return response, prompt
        # check if run llm is set
        if not STSessionManager.llm_ready_to_run():
            response = "I am not real LLM. Check the \" Run LLM\" option to rum LLM"
            return response, prompt

        if not STSessionManager.is_llm_manager_set() or not STSessionManager.get_llm_manager().llm:
            # print(STSessionManager.get_api_key())
            llm = LLMManager.get_llm(STSessionManager.get_api_key())
            # print(llm)
            llm_manager = LLMManager(llm)
            STSessionManager.set_llm_manager(llm_manager)

        llm_manager = STSessionManager.get_llm_manager()
        print(f"LLM magager called: {llm_manager.llm}")
        try:
            response = llm_manager.llm(prompt)
        except Exception as e:
            response = e

        return response, prompt
