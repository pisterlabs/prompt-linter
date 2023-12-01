from langchain.schema.runnable import Runnable
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer, CodeBertTokenizer, MBartTokenizer

class TokenizerRunnable(Runnable):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def run(self, text):
        return self.tokenizer.encode(text)

class PreprocessingRunnerBranch:
    def __init__(self):
        self.tokenizers = {
            'bert': TokenizerRunnable(BertTokenizer.from_pretrained('bert-base-uncased')),
            'gpt2': TokenizerRunnable(GPT2Tokenizer.from_pretrained('gpt2')),
            'roberta': TokenizerRunnable(RobertaTokenizer.from_pretrained('roberta-base')),
            'codebert': TokenizerRunnable(CodeBertTokenizer.from_pretrained('codebert-base')),
            'mbart': TokenizerRunnable(MBartTokenizer.from_pretrained('facebook/mbart-large-cc25'))
        }
    
    def tokenize(self, text, tokenizer_key):
        tokenizer_runnable = self.tokenizers.get(tokenizer_key)
        if tokenizer_runnable:
            return tokenizer_runnable.run(text)
        else:
            raise ValueError(f"Unknown tokenizer key: {tokenizer_key}")

# Usage:
preprocessing_runner_branch = PreprocessingRunnerBranch()
tokenized_text = preprocessing_runner_branch.tokenize("some text or code", 'bert')  # Example usage with BERT tokenizer