import cohere as mainCohere
import os
from dotenv import load_dotenv

load_dotenv()

co = mainCohere.Client(os.getenv('api_key'))

#@title Create the prompt (Run this cell to execute required code) {display-mode: "form"}

class cohereExtractor():
    def __init__(self, examples, example_labels, labels, task_desciption, example_prompt):
        self.examples = examples
        self.example_labels = example_labels
        self.labels = labels
        self.task_desciption = task_desciption
        self.example_prompt = example_prompt

    def make_prompt(self, example):
        examples = self.examples + [example]
        labels = self.example_labels + [""]
        return (self.task_desciption +
                "\n---\n".join( [examples[i] + "\n" +
                                self.example_prompt + 
                                 labels[i] for i in range(len(examples))]))

    def extract(self, example):
      extraction = co.generate(
          model='command-nightly',
          prompt=self.make_prompt(example),
          max_tokens=15,
          temperature=0.5,
          stop_sequences=["\n"])
      return(extraction.generations[0].text)
    
name_examples = [
("Jerry", "Can you give me a water status update on Jerry?"),
("Steven", "How is my plant Steven doing?"),
("Steven", "How is Steven doing?"),
("Waleed", "Water Waleed please."),
("Hack the North", "Did I over water Hack the North?"),
("Bento", "Is Bento too hot in the sun?"),
]

def extractName(input):
    cohereNameExtractor = cohereExtractor([e[1] for e in name_examples], [e[0] for e in name_examples], [], "", "extract the name from the command or question:")

    try:
        extracted_text = cohereNameExtractor.extract(input)
        return extracted_text
    except Exception as e:
        print('ERROR: ', e)