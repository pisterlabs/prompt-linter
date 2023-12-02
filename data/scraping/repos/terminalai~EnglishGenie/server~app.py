from flask import Flask, request
from flask_cors import CORS
from sgnlp.models.csgec import (
    CsgConfig,
    CsgModel,
    CsgTokenizer,
    CsgecPreprocessor,
    CsgecPostprocessor,
    download_tokenizer_files,
)
from sgnlp.models.coherence_momentum import CoherenceMomentumModel, CoherenceMomentumConfig, \
    CoherenceMomentumPreprocessor
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

app = Flask(__name__)
CORS(app)

config = CsgConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/csgec/config.json")
model = CsgModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/csgec/pytorch_model.bin",
    config=config,
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/src_tokenizer/",
    "csgec_src_tokenizer",
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/ctx_tokenizer/",
    "csgec_ctx_tokenizer",
)
download_tokenizer_files(
    "https://storage.googleapis.com/sgnlp/models/csgec/tgt_tokenizer/",
    "csgec_tgt_tokenizer",
)
src_tokenizer = CsgTokenizer.from_pretrained("csgec_src_tokenizer")
ctx_tokenizer = CsgTokenizer.from_pretrained("csgec_ctx_tokenizer")
tgt_tokenizer = CsgTokenizer.from_pretrained("csgec_tgt_tokenizer")

preprocessor = CsgecPreprocessor(src_tokenizer=src_tokenizer, ctx_tokenizer=ctx_tokenizer)
postprocessor = CsgecPostprocessor(tgt_tokenizer=tgt_tokenizer)

config1 = CoherenceMomentumConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/config.json"
)
model1 = CoherenceMomentumModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/coherence_momentum/pytorch_model.bin",
    config=config1
)

preprocessor1 = CoherenceMomentumPreprocessor(config1.model_size, config1.max_len)

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-finetuned-squadv2",
    tokenizer="mrm8488/spanbert-finetuned-squadv2"
)

qa_pipeline2 = pipeline(
    model="mrm8488/t5-base-e2e-question-generation"
)


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model2 = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
kw_model = KeyBERT()

@app.route('/grammar', methods=['GET'])
def grammar():
    text = request.args.get("text")
    batch_source_ids, batch_context_ids = preprocessor([text])
    predicted_ids = model.decode(batch_source_ids, batch_context_ids)
    predicted_texts = postprocessor(predicted_ids)
    return predicted_texts

@app.route("/coherence", methods=["GET"])
def coherence():
    text = request.args.get("text")
    text1_tensor = preprocessor1([text])

    text1_score = model1.get_main_score(text1_tensor["tokenized_texts"]).item()

    return str(text1_score)

@app.route("/answer", methods=["GET"])
def answer():
    passage = request.args.get('passage')
    question = request.args.get('question')
    print(passage, question)
    return qa_pipeline({'context': str(passage),'question': str(question)})['answer']

@app.route("/similarity", methods=["GET"])
def similarity():
    userText = request.args.getlist('userText')
    answerText = request.args.getlist('answerText')
    userText = kw_model.extract_keywords(userText, keyphrase_ngram_range=(1,1), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=10)
    userText = ' '.join([x for x, y in userText])
    answerText = kw_model.extract_keywords(answerText, keyphrase_ngram_range=(1,1), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=10)
    answerText = ' '.join([x for x, y in answerText])
    sentences = [userText, answerText]
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model2(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()
    return str(cosine_similarity([mean_pooled[0]], mean_pooled[1:])[0][0])

@app.route("/question", methods=["GET"])
def question():
    text = request.args.get("text")
    question = qa_pipeline2(text)
    return question[0]['generated_text']

if __name__ == '__main__':
    app.run(port=5000)
