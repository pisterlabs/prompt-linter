import pandas as pd
import cohere
from sklearn.model_selection import train_test_split
import joblib



def likes_para_categorico(likes:int)-> str: #transforma a coluna de likes em uma coluna de strings categoricas 
    if likes < 150000:
        return "nao viral"
    else:
        return "viral"
     
df = pd.read_csv("pipeline_e_datasets/dados_tratados1.csv") #ler dataset

Y_label = df['like_count']
Y_label = Y_label.apply(likes_para_categorico) #label com os dados corretos
X_label = df.drop(columns=["like_count"]) #tiramos a coluna das labels das features

sentences_train, sentences_test, labels_train, labels_test = train_test_split(
            X_label, Y_label, test_size=0.25, random_state=2) #separar dataset de treino e teste

api_key = "*****"  #chave api cohere

co = cohere.Client(api_key)

sentences_train = sentences_train.iloc[:, 0].tolist()  #transforma os dataframes em listas de strings para o embeddings
sentences_test = sentences_test.iloc[:, 0].tolist()

embeddings_train = co.embed(texts=sentences_train,
                             model="embed-english-v2.0").embeddings #embeddings cohere

embeddings_test = co.embed(texts=sentences_test,
                             model="embed-english-v2.0").embeddings

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


svm_classifier = make_pipeline(StandardScaler(), SVC(class_weight='balanced',probability=True))  #pipeline com o scaler de dados e o modelo SVC

svm_classifier.fit(embeddings_train, labels_train) # treina o modelo com os embeddings da cohere

joblib.dump(svm_classifier, 'svm_classifier_model2.pkl') #salva o modelo


score = svm_classifier.score(embeddings_test, labels_test) #score do modelo

def classificador_textos(texto:str, modelo_ML)-> dict: #função que retona a classificação e a probabilidade de uma string estar em uma certa categoria
    embeddings =co.embed(texts=[texto],
                             model="embed-english-v2.0").embeddings
    
    dict_retorno: dict ={}
    arr_probabilidade: list = modelo_ML.predict_proba(embeddings)

    if arr_probabilidade[0][0] > arr_probabilidade[0][1]:
        dict_retorno['previsão'] = "não viral"
        dict_retorno['probabilidade'] = arr_probabilidade[0][0]
    else:
        dict_retorno['previsão'] = "viral"
        dict_retorno['probabilidade'] = arr_probabilidade[0][1]

    return dict_retorno

    
print(f"Validation accuracy on Large is {100*score}%!") #printa o score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = svm_classifier.predict(embeddings_test)

#print(classification_report(labels_test, y_pred))

with open("classificacao.csv", "w") as f:
    f.write(classification_report(labels_test, y_pred)) #escreve CSV de classificacao

matrix = confusion_matrix(labels_test, y_pred)  #cria matriz de correlação
plt.figure(figsize=(10,7))
ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_classifier.classes_,
            yticklabels=svm_classifier.classes_)

ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_facecolor('black')
ax.figure.set_facecolor('black')

ax.set_title("matriz de confusao", color= "white")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.savefig("matriz.png")

