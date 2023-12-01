import adventure as adv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import __main__
import random
import openai
import os


openai.api_key = os.getenv('OPENAI_API_KEY')


class BinaryOpenAINN(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack_with_sigmoid = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack_with_sigmoid(x)
        return logits


class MultilabelOpenAINN(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack_with_sigmoid = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack_with_sigmoid(x)
        return logits


class BinaryNeuralNetwork(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack_with_sigmoid = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack_with_sigmoid(x)
        return logits


class BinaryWith256NeuralNetwork(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack_with_sigmoid = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack_with_sigmoid(x)
        return logits


class MultiLabelNeuralNetwork(nn.Module):  # the NN with linear relu layers and one sigmoid in the end
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class GAME:  # a running game containing its adventure, its PC, the current status (stage), etc.
    def __init__(self, adventure, use_roberta=True):
        self.npc_object = None
        self.object = None
        self.answer = None
        self.adventure = adventure
        self.message = ''
        self.response = None
        self.stage = adventure.starting_stage
        self.triggering = None
        self.trigger_count = 0
        self.use_roberta = use_roberta
        if self.use_roberta:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
            self.embed_model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
        # At some point: consider take-back option; ideally a full history which isn't too long to save including a log

    def __call__(self, message):
        self.message = message
        self.check_for_trigger()
        if self.response is not None:
            return self.response
        self.generate_response()
        self.check_for_secret()
        self.response = f'You are here: {self.stage["location"].name}\n\n{self.response}'
        if self.get_by_name('won').value:
            self.response = f'{self.response}\n\nYou have successfully finished this adventure. ' \
                            f'You can return to the main menu with "/exit".'
        return self.response

    def check_for_secret(self):
        self.response = self.response
        secs = self.stage['location'].secrets
        for i in self.stage['npcs']:
            secs = secs + i.secrets
        secrets = []
        for i in secs:
            if i.activation_flag.value and not i.found:
                secrets.append(i)
        just_found = []
        for i in secrets:
            a = f'Secret: ""{i.name}. {i.prompt}""\nResponse: ""{self.response}""'
            if self.use_roberta:
                setattr(__main__, "NeuralNetwork", BinaryNeuralNetwork)
                model = torch.load('AI_stuff/Moduls/secrets/model.pt')
                model.eval()
                a = self.long_roberta(a)
                x = round(model(a).item())
            else:
                setattr(__main__, "NeuralNetwork", BinaryOpenAINN)
                model = torch.load('models/secret_model.pt')
                model.eval()
                a = openai.Embedding.create(input=a, model="text-embedding-ada-002")["data"][0]["embedding"]
                x = round(model(torch.tensor(a)).item()+0.25)
            if x == 1:
                just_found.append(i)
                i.found = True
        for i in just_found:
            self.response += f'\n\nYou found out the following information: {i.prompt}'
        for i in self.adventure.flags:
            i.check(self.stage)

    def check_for_trigger(self):
        if self.triggering is not None:
            self.trigger_count += 1
            self.response = self.triggering(self)
            if self.response is not None:
                return
            self.triggering = None
            self.trigger_count = 0
        for i in self.adventure.trigger:
            i.call_flag.check(self.stage)  # just in case there are dependencies on other flags
            if i.activation_flag.value and i.call_flag.value:
                self.triggering = i
                self.response = self.triggering.call(self)
                if self.response is not None:
                    return
                self.triggering = None
                self.trigger_count = 0
        self.response = None  # this line is necessary to overwrite the prior response
        for i in self.adventure.flags:
            i.check(self.stage)
        return

    def generate_response(self):
        # possible objects are necessary; possible objects are in self.stage
        poss_objs = [self.stage['location']] + self.stage['npcs']
        prompt = ''
        for i in range(len(poss_objs)):
            prompt += f'{i+1}. {poss_objs[i].name}\n'
        prompt += f'\n{self.message}'
        if self.use_roberta:
            setattr(__main__, "NeuralNetwork", MultiLabelNeuralNetwork)
            model = torch.load('AI_stuff/Moduls/object/model.pt')
            model.eval()
            a = self.long_roberta(prompt)
            pred = model(a)
            # pred = pred[0]
        else:
            setattr(__main__, "NeuralNetwork", MultilabelOpenAINN)
            model = torch.load('models/object_model.pt')
            model.eval()
            a = openai.Embedding.create(input=prompt, model="text-embedding-ada-002")["data"][0]["embedding"]
            pred = model(torch.tensor(a))
        x = torch.argmax(pred[0:len(poss_objs)]).item()
        self.object = poss_objs[x]
        if len(poss_objs) > 1:
            y = torch.argmax(pred[1:len(poss_objs)]).item()
            self.npc_object = poss_objs[1:][y]
        else:
            self.npc_object = None
        poss_input_type = ['info', 'verbatim', 'action', 'fight', 'room change']
        if self.use_roberta:
            setattr(__main__, "NeuralNetwork", MultiLabelNeuralNetwork)
            model = torch.load('AI_stuff/Moduls/type/model.pt')
            model.eval()
            a = self.long_roberta(self.message)
            pred = model(a)
        else:
            setattr(__main__, "NeuralNetwork", MultilabelOpenAINN)
            model = torch.load('models/type_model.pt')
            model.eval()
            a = openai.Embedding.create(input=self.message, model="text-embedding-ada-002")["data"][0]["embedding"]
            pred = model(torch.tensor(a))
        # print(pred)
        x = torch.argmax(pred).item()
        input_type = poss_input_type[x]
        if self.message.startswith('/') or self.message.startswith('\\'):
            self.message = self.message.replace('\\', '/')
            if self.message.startswith('/i'):
                input_type = 'info'
            elif self.message.startswith('/v'):
                input_type = 'verbatim'
            elif self.message.startswith('/a'):
                input_type = 'action'
            elif self.message.startswith('/f'):
                input_type = 'fight'
            elif self.message.startswith('/r'):
                input_type = 'room change'
        types = {'info': self.info,
                 'verbatim': self.verbatim,  # only with npcs
                 'action': self.action,
                 'fight': self.fight,  # only with npcs
                 'room change': self.location_change}
        types[input_type]()

    # the following functions may seem a little repetitive, but I believe it is easier to implement new concepts like
    # abilities if they are written like this.
    def info(self):
        if isinstance(self.object, adv.LOCATION):
            self.response = self.object(self.message, self.stage['npcs'])
        else:
            self.response = self.object(self.message)

    def verbatim(self):
        if self.npc_object is not None:
            self.response = self.npc_object.speak(self.message)
        else:
            self.response = 'You are talking to an empty room.\n'
            self.response += self.object(self.message, self.stage['npcs'])

    def fight(self):
        if self.npc_object is not None:
            self.response = self.npc_object.fight(self.message)
        else:
            self.response = 'You are attacking an empty room.\n'
            self.response += self.object(self.message, self.stage['npcs'])

        # At some point: when fighting you could lose objects instead of health. And gain objects or even secrets.

    def action(self):
        if isinstance(self.object, adv.LOCATION):
            self.response = self.object(self.message, self.stage['npcs'])
        else:
            self.response = self.object(self.message)
        # At some point: Let actions have effect. Like picking up objects

    def location_change(self):
        poss_where = []
        a = ''
        for i in self.adventure.locations:
            if i.activation_flag.value:
                poss_where.append(i)
                a += f'{self.adventure.locations.index(i)+1}. {i.name}\n'
        a += f'\n{self.message}'
        if self.use_roberta:
            setattr(__main__, "NeuralNetwork", BinaryWith256NeuralNetwork)
            model = torch.load('AI_stuff/Moduls/go_to/model.pt')
            model.eval()
            a = self.long_roberta(self.message)
        else:
            setattr(__main__, "NeuralNetwork", BinaryOpenAINN)
            model = torch.load('models/go_to_model.pt')
            model.eval()
            a = openai.Embedding.create(input=self.message, model="text-embedding-ada-002")["data"][0]["embedding"]
            a = torch.tensor([a])
        preds = []
        for i in self.adventure.locations:
            if i.activation_flag.value:
                poss_where.append(i)
                if self.use_roberta:
                    b = self.long_roberta(i.name)
                else:
                    b = openai.Embedding.create(input=i.name, model="text-embedding-ada-002")["data"][0]["embedding"]
                    b = torch.tensor([b])
                pred = model(torch.cat((a[0], b[0])))
                preds.append(pred.item())
        where = poss_where[preds.index(max(preds))]
        if max(preds) < 0.5:
            where = random.choice(poss_where)
        self.stage['location'] = where
        self.stage['npcs'] = self.adventure.rand_npcs()
        self.response = f'You are going to {where.name}.'

    def get_by_name(self, name):
        lst = self.adventure.secrets + self.adventure.locations + self.adventure.npcs + self.adventure.flags
        for i in lst:
            if i.name == name:
                return i
        return None

    def long_roberta(self, sentences):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(roberta_model_output, attention_mask):
            token_embeddings = roberta_model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(
                token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']

        '''
        # Load model from HuggingFace Hub
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
        if model is None:
            model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
        '''

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # test if this works with truncation=False

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
