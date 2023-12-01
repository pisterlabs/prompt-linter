import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics import classification_report, PrecisionRecallDisplay
from openai.embeddings_utils import cosine_similarity, get_embedding

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
CSV_DB_NAME = "Gx_review_embeddings.csv"
THRESHOLD = 0

df = pd.read_csv(CSV_DB_NAME)
df["Embedding"] = df.Embedding.apply(eval).apply(np.array)
df["Sentiment"] = df.Score.replace(
    {1: "Negative", 2: "Negative", 4: "Positive", 5: "Positive"}
)
df = df[["Sentiment", "Summ_and_Text", "Embedding"]]


def evaluate_classification_labels(labels: list[str], model=EMBEDDING_MODEL, threshold=THRESHOLD):
    """
    이 함수는 분류 레이블의 정확도를 테스트하여 정밀도-호출 곡선을 출력합니다. 
    이를 통해 Positive/Negative와 같은 레이블 또는 
    'Positive product review' 및 'Negative product review'와 같은 보다 복잡한 레이블을 테스트하여 
    어떤 것이 Positive/Negative review 임베딩과 가장 잘 일치하는지 확인할 수 있습니다.
    레이블: 두 개의 용어 목록. 첫 번째 용어는 긍정적인 리뷰를 의미하고 두 번째 용어는 부정적인 리뷰를 의미합니다.
    """
    test_label_embeddings = [get_embedding(label, engine=model) for label in labels]

    def label_score(review_emb, test_label_emb):
        positive_similarity = cosine_similarity(review_emb, test_label_emb[0])
        negative_similarity = cosine_similarity(review_emb, test_label_emb[1])
        return positive_similarity - negative_similarity

    probabilities = df["Embedding"].apply(
        lambda review_emb: label_score(review_emb, test_label_embeddings)
    )
    predictions = probabilities.apply(lambda score: "Positive" if score > threshold else "Negative")

    report = classification_report(df["Sentiment"], predictions)
    print(report)
    display = PrecisionRecallDisplay.from_predictions(
        df["Sentiment"], probabilities, pos_label="Positive"
    )
    display.ax_.set_title("Precision-Recall curve for test classification labels")


def add_prediction_to_df(labels: list[str], model=EMBEDDING_MODEL, threshold=THRESHOLD):
    """
    이 함수는 제공된 레이블을 기준으로 예측 열을 데이터 프레임에 추가합니다.
    """
    label_embeddings = [get_embedding(label, engine=model) for label in labels]

    def label_score(review_emb, test_label_emb):
        positive_similarity = cosine_similarity(review_emb, test_label_emb[0])
        negative_similarity = cosine_similarity(review_emb, test_label_emb[1])
        return positive_similarity - negative_similarity

    probabilities = df["Embedding"].apply(
        lambda review_emb: label_score(review_emb, label_embeddings)
    )
    df["Prediction"] = probabilities.apply(lambda score: "Positive" if score > threshold else "Negative")


add_prediction_to_df(["긍정적인 감성의 제품 리뷰", "부정적인 감정을 가진 제품 리뷰"])
pd.set_option('display.max_colwidth', None)
printdf = df.drop(columns=["Embedding"])
print(printdf.head(10))