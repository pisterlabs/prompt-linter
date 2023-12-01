import cohere
import joblib
import os

model = joblib.load(os.path.join("svm_classifier_model2.pkl"))

co = cohere.Client("API_KEY")

def classificador_textos(texto:str)-> dict: #primeiro numero do array de retorno é a probabilidade de ser não viral, o segundo de ser viral
    embeddings =co.embed(texts=[texto],
                             model="embed-english-v2.0").embeddings
     
    dict_retorno: dict ={}
    arr_probabilidade: list = model.predict_proba(embeddings)
    
    if arr_probabilidade[0][0] > arr_probabilidade[0][1]:
        dict_retorno['previsão'] = "não viral"
        dict_retorno['probabilidade'] = arr_probabilidade[0][0]
    else:
        dict_retorno['previsão'] = "viral"
        dict_retorno['probabilidade'] = arr_probabilidade[0][1]

    return dict_retorno
