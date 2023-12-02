from classifier.classifier import Classifier
from langchain.agents import Tool
from memory.memory_writer import write_to_json
from api.file_watcher import FileWatcher
from simple_chalk import chalk, red

class ClassifierManager: 
    def __init__ (self):
        self.has_classifier = False;
        self.tools = [
            Tool(
                name = "Store",
                func = write_to_json,
                description="Use to store important memories"
            )
        ]  
        self.classifier = self.create_classifier()

    def create_classifier(self):
        # Initialize the CustomAgent with any necessary tools
        try:   
            classifier = Classifier()
            self.has_classifier = True
            return classifier
        
        except Exception as e:
            return e

    def start_classifier(self):
        file_watcher = FileWatcher(self.classifier)
        file_watcher.watch_file()

    def print_red(self, text):
        print(chalk.red(text))
        return True

    #def stop_classifier(self):

    def get_classifier(self):
        return self.classifier

    def restart_classifier(self):
        self.classifier.restart()

    def intake(self, conversation):
        print(chalk.red("got HIT!"))
        return self.classifier.intake(conversation)
