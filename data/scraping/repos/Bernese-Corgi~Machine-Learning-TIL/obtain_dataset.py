import pandas as pd
import tiktoken

from openai.embeddings_utils import get_embedding

# ------------------------ embedding model parameters ------------------------ #
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

# -------------------------- load & inspect dataset -------------------------- #
input_datapath = "data/fine_food_reviews_1k.csv"
# pd.read_csv() 함수를 사용하여 input_datapath로 지정한 CSV 파일을 읽어서 데이터프레임 df에 저장합니다. index_col=0 옵션은 첫 번째 열을 인덱스로 사용하도록 지정하는 것입니다.
df = pd.read_csv(input_datapath, index_col=0)
# 데이터프레임 df의 열을 "Time", "ProductId", "UserId", "Score", "Summary", "Text" 열로 제한합니다. 다른 열은 제외됩니다.
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
# 데이터프레임 df에서 결측치가 있는 행을 제거합니다.
df = df.dropna()
# "combined"라는 새로운 열을 추가합니다. 이 열은 "Title: {Summary 값}; Content: {Text 값}" 형식의 문자열로 구성됩니다.
df["combined"] = (
    # df.Summary.str.strip()는 Summary 열의 값의 앞뒤 공백을 제거하고, df.Text.str.strip()는 Text 열의 값의 앞뒤 공백을 제거하는 것을 의미합니다.
    f"Title: {df.Summary.str.strip()}; Content: {df.Text.str.strip()}"
)
# 데이터프레임 df의 처음 2개 행을 출력합니다.
print(df.head(2))

# - subsample to 1k most recent reviews and remove samples that are too long - #
top_n = 1000
# 처음 2,000개 항목으로 첫 번째 컷, 절반 미만이 필터링될 것이라고 가정
# 데이터프레임 df를 "Time" 열을 기준으로 정렬하고, 최근에 작성된 상위 top_n * 2개의 행을 선택합니다.
df = df.sort_values("Time").tail(top_n * 2)
# "Time" 열을 제거합니다.
df.drop("Time", axis=1, inplace=True)

# tiktoken 라이브러리의 get_encoding() 함수를 사용하여 embedding_encoding에 해당하는 인코딩 방식을 가져옵니다. 이 인코딩 방식은 encoding 변수에 저장됩니다.
encoding = tiktoken.get_encoding(embedding_encoding)

# ------------------ omit reviews that are too long to embed ----------------- #
# "combined" 열의 각 행에 대해 인코딩된 토큰 수를 구하고, "n_tokens" 열에 저장합니다.
df['n_tokens'] = df.combined.apply(lambda x: len(encoding.encode(x)))
# "n_tokens" 열 값이 max_token보다 작거나 같은 행만 선택합니다. 상위 top_n개의 행을 선택하여 데이터프레임 df를 업데이트합니다.
df = df[df.n_tokens <= max_tokens].tail(top_n)
print(len(df))

df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("data/fine_food_reviews_with_embeddings_1k.csv")