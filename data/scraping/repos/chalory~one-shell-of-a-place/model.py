import pandas as pd
import cohere
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_colwidth", None)

df = pd.read_csv(
    "https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv",
    delimiter="\t",
    header=None,
)

num_examples = 500
df_sample = df.sample(num_examples)
sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    list(df_sample[0]), list(df_sample[1]), test_size=0.25, random_state=0
)
api_key = "gqVPQNIEYu4Dei3YOaopVg9xwUyWU1VDD7tMBEyq"

co = cohere.Client(api_key)


embeddings_train = co.embed(
    texts=sentences_train, model="large", truncate="LEFT"
).embeddings
embeddings_test = co.embed(
    texts=sentences_test, model="large", truncate="LEFT"
).embeddings


svm_classifier = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))

svm_classifier.fit(embeddings_train, labels_train)

# test = ["it'\s okay", "It was horrible", "absolutely disgusting"]
test = ["I like turtles", "They are nice", "good"]

embeddings_test1 = co.embed(texts=test, model="large", truncate="LEFT").embeddings

score = svm_classifier.predict(embeddings_test1)
print(score)
# print(f"Validation accuracy on Large is {100*score}%!")
