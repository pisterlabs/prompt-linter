import argparse
import os
import tempfile
import tarfile

import torch
import torch.nn.functional as F

from itertools import chain
from pytorch_pretrained_bert import cached_path
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

# GLOBALS
PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

def download_pretrained_model():
    """ Download and extract finetuned model from S3
    
    Returns:
        str -- tempdir: filepath (possibly cached) for loading pre-trained model
    """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    # logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    print("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    
    return tempdir

class HuggingFaceModel(object):
    def __init__(self, decode_method:str):
        """ Inference wrapper class for getting predictions from model
        and decoding into text to be presented to end-user

        Arguments:
            tokenizer -- Instance of pytorch_pretrained_bert.OpenAIGPTTokenizer
            model_path {str} -- Path to trained model
        """
        
        self.no_sample = True if decode_method == 'greedy' else False
        self.SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
        self.max_len = 45
        self.device = "cpu"
        self.max_history = 2
        self.history = []
        self._load_model()

    def _load_model(self):
        """ Helper function for loading model and tokenizer in one shot
        and assigning as class attributes

        """
        # Load tokenizer and model within `main` function
        ckpt = download_pretrained_model()
        print("Model location:", ckpt)
        
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(ckpt)
        self.model = OpenAIGPTLMHeadModel.from_pretrained(ckpt)
        print("Tokenizer and model loaded...")

    def _encode_personality(self, personality_list:list):
        """ Internal function for encoding "personality" attributes
        for use in model predictions.
        
        Arguments:
            personality_list {list} -- List of strings encoding personality traits.
        
        Returns:
            list -- List of ints containing word ID's for encoded personality traits.
            
        """
        personality_encoded = [self.tokenizer.encode(p) for p in personality_list]
        return personality_encoded

    def _build_input_from_segments(self, persona, history, reply, lm_labels=False, with_eos=True):
        """ Build a sequence of input from 3 segments: persona, history and last reply 
        
        Arguments:
            persona {list} -- List of persona tokens
            history {list} -- List of previous utterances in dialog
            reply {list} -- List of reply tokens
        
        Keyword Arguments:
            lm_labels {bool} -- Switch indicating whether labels are present (for eval metrics) (default: {False})
            with_eos {bool} -- Switch indicating whether pre-processing should be aware of special <end-of-sentence> token (default: {True})
        
        Returns:
            [type] -- [description]
        """
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[:-1])

        instance = {}
        sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance, sequence

    def top_filtering(self, logits, top_k:int=0, top_p:float=0.0, threshold:float=-float('Inf'), filter_value:float=-float('Inf')):
        """Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        
        Arguments:
            logits {torch.Tensor} -- [logits distribution shape (vocabulary size)]
        
        Keyword Arguments:
            top_k {int} -- [<=0: no filtering, >0: keep only top k tokens with highest probability.] (default: {0})
            top_p {float} -- [<=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.] (default: {0.0})
            threshold {float} -- [a minimal threshold to keep logits] (default: {-float('Inf')})
        
        Returns:
            logits {torch.Tensor} -- [torch Tensor containing weights subjected to filtering as outlined in function]
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            print('orig logits:', logits.size())
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            print("sorted_logits:", sorted_logits.size())
            print("sorted_indices:", sorted_indices.size())
            print('cumulative_probs:', cumulative_probabilities.size())

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            print('sorted_indices_to_remove:', sorted_indices_to_remove.size())
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            print('sorted_indices_to_remove:', sorted_indices_to_remove.size())

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

    def sample_sequence(self, personality:list, history:list, current_output=None):
        """ Loop through inputs and generated outputs to generate
        the next output token in a response sequence
        
        Arguments:
            personality {list} -- [List of personality tokens]
            history {list} -- [List of tokens that contain the history of conversation up to the current turn]
        
        Keyword Arguments:
            current_output {list or None} -- [Potential list of current output to be considered in generating future output] (default: {None})
        
        Returns:
            current_output {list} -- [List containing output generated up to current timestep]
        """
        # Hard-coded
        top_k = 0
        top_p = 0.9
        no_sample = False
        min_length = 5
        max_length = 20
        temperature = 0.7

        assert (self.model is not None) and (self.tokenizer is not None)
        
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        if current_output is None:
            current_output = []

        for i in range(max_length):
            instance, sequence = self._build_input_from_segments(personality, history, current_output, self.tokenizer, with_eos=False)
            # print("instance from sample_sequence function:")
            # print(instance)

            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)

            logits = self.model(input_ids, token_type_ids=token_type_ids)
            print('logits from inside sample_sequence:', logits.size())

            logits = logits[0, -1, :] / temperature
            print('in_logits:', logits.size())
            logits = self.top_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
            if i < min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        return current_output

    def get_response(self, input_seq:str, personality:list=[]):
        """ Wrapper method for taking in dialog input as a string, preprocessing, 
        feeding to model, and generating output from the model.
        
        Arguments:
            input_seq {str} -- [Most recent input of the conversation supplied as a string]
        
        Keyword Arguments:
            personality {list} -- [Personality tokens supplied in list (can be empty list)] (default: {[]})
        
        Returns:
            decoded_string {str} -- [Decoded string response from model]
        """
        self.model.to(self.device)
        self.model.eval()
        
        torch.random.manual_seed(7)
        torch.cuda.manual_seed(7)

        if len(personality) == 0:
            personality = ["i like playing football.", "i am from NYC."]

        personality_encoded = self._encode_personality(personality)

        self.history.append(self.tokenizer.encode(input_seq))

        with torch.no_grad():
            out_ids = self.sample_sequence(personality_encoded, self.history)
        self.history.append(out_ids)
        decoded_string = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        return decoded_string

    def run_interactive(self):
        """ Testing method to run model in "online" way to continually get user input in real-time
        and generate output in interactive session 
        """
        self.model.to(self.device)
        self.model.eval()
        
        torch.random.manual_seed(7)
        torch.cuda.manual_seed(7)
        
        personality = ["i like playing football.", "i am from NYC."]
        personality_encoded = self._encode_personality(personality)
        
        # history = []
        while True:
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            history.append(self.tokenizer.encode(raw_text))
            with torch.no_grad():
                out_ids = self.sample_sequence(personality_encoded, self.history)
            self.history.append(out_ids)
            self.history = self.history[-(2*self.max_history+1):]
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            print(out_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode_method', type=str, required=False, default='sample')
    parser.add_argument('--gpu', type=int, required=False, default=-1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model_obj = HuggingFaceModel(decode_method=args.decode_method)
    # model_obj.run_interactive()

    samples = ["Hey, how's it going?"] # ["Hey, how's it going?", "Give me your credit card information!", "I think your account has been hacked."]
    for s in samples:
        r = model_obj.get_response(input_seq=s)
        print()
        print(s)
        print('==>', r)
        print()

    print("SUCCESSFULLY GOT OUTPUT")
