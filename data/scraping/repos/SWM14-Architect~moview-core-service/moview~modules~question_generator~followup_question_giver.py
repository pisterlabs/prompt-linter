import openai
from moview.utils.prompt_loader import PromptLoader
from moview.environment.llm_factory import LLMModelFactory
from moview.config.loggers.mongo_logger import prompt_result_logger
from moview.decorator.retry_decorator import retry
from moview.utils.singleton_meta_class import SingletonMeta
from moview.utils.mixin.directory_mixin import DirectoryMixin


class FollowUpQuestionGiver(DirectoryMixin, metaclass=SingletonMeta):

    def __init__(self, prompt_loader: PromptLoader):
        self.prompt = prompt_loader.load_prompt_json(FollowUpQuestionGiver.__name__)
        openai.api_key = LLMModelFactory.load_api_key_for_open_ai()

    @retry()
    def give_followup_question(self, question: str, answer: str) -> str:
        """
        꼬리질문을 출제하는 메서드

        Args:
            question: 현재 질문
            answer: 현재 질문에 대한 답변

        Returns:
            출제할 꼬리 질문

        """

        model = "gpt-3.5-turbo-16k"

        messages = [{
            "role": "system",
            "content": self.prompt.format()
        }, {
            "role": "user",
            "content": f"""
                    면접관의 질문: {question}

                    면접 지원자의 답변: {answer} 
                    
                    양식을 지켜서 후속 질문을 생성하세요.    
                    """
        }]

        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.3)

        prompt_result = response['choices'][0]['message']['content']

        prompt_result_logger("followup question prompt result", prompt_result=prompt_result)

        return prompt_result
