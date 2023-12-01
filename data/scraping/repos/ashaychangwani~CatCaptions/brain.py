from gpt_client import GPTClient
from cohere_client import CohereClient
from description_gen import DescriptionGen
class Brain:
    def __init__(self):
        self.gpt_client = GPTClient()
        print('GPT Client Ready')
        self.cohere_client = CohereClient()
        print('Cohere Client Ready')
        self.description_generator = DescriptionGen()
        print('Description Client Ready')
        
    
    def process(self, image_stream, model = 'GPT'):
        description = self.description_generator.generate_description(image_stream)
        print('Description for image',description)
        attempts = 0
        captions = None
        if model == 'GPT': 
            while captions is None and attempts < 10:
                try:
                    captions = self.gpt_client.create_captions(description)
                except:
                    attempts+=1
        else:
            while captions is None and attempts < 10:
                try:
                    captions = self.cohere_client.create_captions(description)
                except:
                    attempts+=1
        print('Captions for image',captions)
        if captions is None:
            return None
        captions = captions.strip().split('\n')
        result = []
        for line in captions:
            if line.find('"') < 0:
                continue
            line = line[line.find('"')+1:]
            line = line[:line.find('"')]
            result.append(line)
            if len(result) == 5:
                break
        return result
        