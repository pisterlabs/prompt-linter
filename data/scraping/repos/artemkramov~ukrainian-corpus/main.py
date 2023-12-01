from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from coreference_ua import CoreferenceUA
from coherence_ua import CoherenceModel
import noun_phrase_ua
app = Flask(__name__)
CORS(app)

print('Init coreference model...')
model_coreference = CoreferenceUA()
print('Init coherence model...')
model_coherence = CoherenceModel()
model_coherence.set_embedder(model_coreference.policy.semantic_embedding.model)
print('Init phrase extractor...')
model_phrase = noun_phrase_ua.NLP()

print('Ready to accept queries')

if __name__ == "__main__":
    app.run(host='0.0.0.0')

def get_text_from_request():
    content = request.json
    if (not (content is None)) and 'text' in content:
        return content['text']
    abort(400)
    
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/get_coreferent_clusters', methods=['POST'])
def get_coreferent_clusters():
    text = get_text_from_request()
    return jsonify(model_coreference.set_text(text).extract_phrases())
    
@app.route('/api/get_phrases', methods=['POST'])
def get_phrases():
    text = get_text_from_request()
    summary = model_phrase.extract_entities(text)
    summary['entities'] = list(summary['entities'])
    return jsonify(summary)
    
@app.route('/api/get_coherence', methods=['POST'])
def get_coherence():
    text = get_text_from_request()
    sentences = model_coherence.ud_model.get_tokens(text)
    sentences = [" ".join(sentence) for sentence in sentences]
    summary = {
        "series": [str(item) for item in list(model_coherence.get_prediction_series(text).flatten())],
        "coherence_product": str(model_coherence.evaluate_coherence_as_product(text)),
        "coherence_threshold": str(model_coherence.evaluate_coherence_using_threshold(text, 0.5)),
        "sentences": sentences
    }
    # print(summary)
    return jsonify(summary)
    