from .ml_models.leafbase_generator import LeafBaseMCQGenerator
from .ml_models.sumbase_generator import SumBaseMCQGenerator
# from .ml_models.open_ai import OpenAIGenerator

from collections import defaultdict

DEFAULT_MODEL_NAME = 'leafQad_base'

class MCQSelector():
    def __init__(self, active: dict) -> None:
        self.active = defaultdict(str, active)
        self.models = {
            'leafQad_base': LeafBaseMCQGenerator() if self.active['leafQad_base'] else None,
            'sumQd_base': SumBaseMCQGenerator() if self.active['sumQd_base'] else None,
            # 'text-davinci-003': OpenAIGenerator() if self.active['text-davinci-003'] else None
        }
    
    def generate_mcq_questions(self, context, model_name, count):
        if model_name in self.models and self.active[model_name]:
            return self.models[model_name].generate_mcq_questions(context, count)
        else:
            return [False]*3
        