from typing import List, Tuple, Optional
from openai_api.task_request.task_base_request import TaskBaseRequest
from openai_api.text_generation.text_generation_request import TextGenerationRequestType, TextGenerationRequest
from openai_api.models import Model


class ClassificationDomain:
    """
    어떤 classification인지 명시하는 클래스입니다.
    
    YNAT 데이터셋을 사용한다면, YNAT 데이터에 관한 label을 입력해야 합니다.
    """
    def __init__(self, labels: List[str]):
        self.labels = labels
    
    def string_to_label_idx(self, string: str) -> int:
        if string not in self.labels:
            raise ValueError(f"string must be in {self.labels}, but {string} is given.")
        return self.labels.index(string)
    
    def label_idx_to_string(self, label_idx: int) -> str:
        if not (0 <= label_idx < len(self.labels)):
            raise ValueError(f"label_idx must be in range [0, {len(self.labels)}), but {label_idx} is given.")
        return self.labels[label_idx]


class ClassificationRequest(TaskBaseRequest):
    """
    classification 문제에 대한 답변을 생성하는 클래스입니다.
    
    args:
        domain: 어떤 classification인지 명시하는 클래스 (label의 종류 등의 정보)
        type: 어떤 방식으로 답변을 생성할지 명시하는 열거형 (chat completion인지 completion인지)
        model: 어떤 모델을 사용할지 명시하는 열거형
        prompt_format: prompt의 형식을 명시하는 문자열
    """
    def __init__(
        self, 
        domain: ClassificationDomain,
        type: TextGenerationRequestType,
        model: Model,
        prompt_format: str,
    ):
        if prompt_format.find("{}") == -1:
            raise ValueError("prompt_format must contain '{}'.")
        
        self.domain = domain
        self.api = TextGenerationRequest.get_api(type=type, model=model)
        self.model = model
        self.prompt_format = prompt_format
    
    def request(self, input: str) -> Tuple[Optional[int], str, str]:
        """
        입력에 대한 분류 결과를 생성합니다.
        
        args:
            input: 분류할 입력
        
        returns:
            Optional[int]: 분류 결과 index
            str: 사용한 prompt
        """
        
        prompt = self.prompt_format.format(input)
        response = self.api.request(prompt=prompt)
        try:
            return self.domain.string_to_label_idx(response), prompt, response
        except ValueError:
            return None, prompt, response
