import os
from abc import ABC, abstractmethod
import json
import numpy
import openai
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from lib.env import OPEN_AI_API_KEY


class OpenAITrainer:
    def __init__(self, model_name, model_params=None):
        self.model_name = model_name
        self.api_key = os.environ.get("OPEN_AI_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        self.model_params = model_params

    def fine_tune(self, training_data):
        # This is a placeholder function. In practice, this would involve
        # sending a POST request to OpenAI's fine-tuning API with the required data.
        fine_tune_url = f"https://api.openai.com/v1/models/{self.model_name}/fine-tune"
        response = requests.post(fine_tune_url, headers=self.headers, json={
            "training_data": training_data,
            **self.model_params
        })
        return response.json()


class Block(ABC):
    def __init__(self, name: str, previous_block_names: list[str] | str = None,
                 next_block_names: list[str] | str = None):
        if name is None or name == '' or name.isspace():
            raise ValueError("Block name invalid, it should be a non-empty string .")

        self.name = name
        if isinstance(previous_block_names, str):
            previous_block_names = [previous_block_names]
        if isinstance(next_block_names, str):
            next_block_names = [next_block_names]

        self.previous_block_names = previous_block_names or []
        self.next_block_names = next_block_names or []

    @abstractmethod
    def forward(self, *inputs):
        pass

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __gt__(self, other):
        if isinstance(other, list):
            if all(isinstance(block, Block) for block in other):
                for block in other:
                    self.__gt__(block)
                return other
        if not isinstance(other, Block):
            raise TypeError("Operand must be an instance of Block.")
        self.next_block_names.append(other.name)
        other.previous_block_names.append(self.name)
        return other

    def __hash__(self):
        return hash(self.name)


class Transformer(Block):
    def __init__(self, name: str, output_json: str, model_name: str):
        super().__init__(name)
        self.model_name = model_name
        try:
            with open(output_json, 'r') as f:
                self.output_structure = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("The output_json file is not a valid JSON file.")

    @abstractmethod
    def forward(self, *inputs):
        pass

    def _create_prompt(self, *inputs):
        # Logic to construct the prompt using inputs and output_structure
        # This is simplified for illustration purposes
        combined_input = "\n\n".join(inputs)
        input_replace: str = self.output_structure['prompt'].replace("INPUT", combined_input)
        if self.output_structure.get("document_structure") is None:
            return input_replace
        return input_replace.replace("OUTPUT", str(self.output_structure['document_structure']))

    @abstractmethod
    def get_trainer(self, **args):
        pass

    def update_model(self, new_model_name_or_identifier, ):
        self.model_name = new_model_name_or_identifier


class LocalTransformer(Transformer):
    def __init__(self, name: str, output_json: str, model_name: str = "google/flan-t5-small",
                 examples: list[(str, str)] = None, bnb_config = None, ia3_config = None, accelerator = None):
        super().__init__(name, output_json, model_name)


        self.model_name = model_name
        if bnb_config == None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = prepare_model_for_kbit_training(AutoModelForCausalLM.from_pretrained(model_name, 
                                                                                              quantization_config = bnb_config))
        if ia3_config != None:
            self.model = get_peft_model(self.model, ia3_config)
        if accelerator != None:
            self.model = accelerator.prepare_model(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        
        self.examples = examples or []

    def forward(self, *inputs):
        prompt = self._create_prompt(*inputs)
        print(prompt)
        # Local model logic
        # (Assuming the inputs are compatible with the model's expected input format)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _create_prompt(self, *inputs):
        prompt = super()._create_prompt(*inputs)
        if self.examples:
            examples_text = "\n".join(
                [f"Example input : {inp} \n Example output : {out}" for (inp, out) in self.examples]) + "\n\n"
            prompt = examples_text + prompt

        return prompt

    def get_trainer(self, train_dataset, eval_dataset=None, training_args=None, compute_metrics=None, ia3_config = False):
        # Assume the `model` attribute holds an instance of a Hugging Face model that's compatible with their Trainer
        # class `training_args` should be an instance of Hugging Face's `TrainingArguments`
     
        
        if training_args is None:
            training_args = TrainingArguments(output_dir='./results', num_train_epochs=3)
        if ia3_config is None:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics
            )
        else:
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                peft_config=ia3_config,
                compute_metrics=compute_metrics
            )
                

        return trainer

    def update_model(self, new_model_name_or_identifier):
        # Logic to update the model in the block after src
        super().update_model(new_model_name_or_identifier)
        self.model = AutoModelForCausalLM.from_pretrained(new_model_name_or_identifier)
        self.tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_identifier)


class OpenAITransformer(Transformer):
    def __init__(self, name: str, output_json: str, model_name: str):
        super().__init__(name, output_json, model_name)
        if not OPEN_AI_API_KEY:
            raise ValueError("OpenAI API key must be provided when using an OpenAI model.")
        openai.api_key = OPEN_AI_API_KEY

    def forward(self, *inputs):
        prompt = self._create_prompt(*inputs)
        # Combine inputs for prompt
        # Make the API call
        try:
            response = openai.Completion.create(
                model=self.model_name,  # Use the fine-tuned model identifier
                prompt=prompt,
                max_tokens=50,  # Adjust as needed
                # Add other parameters like temperature if required for your use case
            )
        except openai.error.OpenAIError as e:
            # Handle any API errors
            print(f"An error occurred: {e}")
            return None

        # Extract and return the generated text from the response
        generated_text = response.choices[0].text.strip() if response.choices else ""
        return generated_text

    def get_trainer(self, model_params):
        # Initialize the OpenAI trainer and start the fine-tuning process
        openai_trainer = OpenAITrainer(self.name, model_params)
        return openai_trainer

    def update_model(self, new_model_name_or_identifier):
        # Logic to update the model in the block after src
        if not OPEN_AI_API_KEY:
            raise ValueError("OpenAI API key must be provided when use_openai_api is True.")
        self.model_name = new_model_name_or_identifier


class SimpleRetriever:
    # Assuming this is a class from an external library
    def __init__(self, resources, name):
        # Initialises TF-IDF from scikit-learn with the resources
        self.vectorizer = TfidfVectorizer()
        self.path = f"data/external/tfidf_{name}.npy"
        self.name = name
        if os.path.exists(self.path):
            self.load()
        else:
            self.matrix = self.vectorizer.fit_transform(resources)
            self.save()
        if isinstance(resources, list):
            self.resources = resources
        elif isinstance(resources, str):
            self.resources = []
            if os.path.isdir(resources):
                for root, dirs, files in os.walk(resources):
                    for file in files:
                        if "jsonl" in file:
                            with open(os.path.join(root, file), 'r') as f:
                                self.resources.extend(f.readlines())
        else:
            raise TypeError("Resources must be a list of strings or a path to a directory.")

    def retrieve(self, query, top_n=5):
        # Uses the TF-IDF vectorizer to retrieve the top_n most relevant resources
        # Compute cosine similarity between the new doc and all existing docs
        new_doc_tfidf = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(new_doc_tfidf, self.matrix).flatten()
        # Get the top N relevant docs
        top_n_indices = cosine_similarities.argsort()[-top_n:][::-1]
        return [(cosine_similarities[i], self.resources[i]) for i in top_n_indices]

    def save(self):
        # Saves the TF-IDF vectorizer and the matrix to a path
        directory = os.path.dirname(self.path)
        print('DIRECTORYYYY : ', directory)
        if not os.path.exists(self.path):        
            # create the directory
            os.makedirs(directory)
        numpy.save(self.path, self.matrix.toarray())

        vectorizer_path = f"{directory}/{self.name}_vectorizer.pkl"

        # Save the state of the vectorizer
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        # Loads the TF-IDF vectorizer and the matrix from a path
        self.matrix = numpy.load(self.path)
        directory = os.path.dirname(self.path)
        vectorizer_path = f"{directory}/{self.name}_vectorizer.pkl"
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)


class Selector(Block):
    def __init__(self, resources, name, top_n=5):
        super().__init__(name)
        self.retriever = SimpleRetriever(resources, name=name)
        self.top_n = top_n

    def forward(self, *inputs):
        # The inputs would be the query or queries for which we seek relevant resources
        results = []
        for query in inputs:
            results.extend(self.retriever.retrieve(query, self.top_n))
        # We'll return a list of resources, possibly concatenated into a single context string
        return '\n'.join([content for _, content in results])
