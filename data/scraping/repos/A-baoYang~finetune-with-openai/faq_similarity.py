from openai_utils import OpenAIAPI
from openai.embeddings_utils import cosine_similarity
import pandas as pd


if __name__ == "__main__":

    openai = OpenAIAPI()
    openai.login()

    faq_ebd = pd.read_pickle("data/cathay_faq_embeddings.pkl")
    query = ""
    n = 3
    while True:

        query = input("嗨，你可以問我任何關於網銀的事情；如果你想要結束對話可以輸入 over 來關閉服務\n")
        if query == "over":
            break

        query_ebd = openai.get_embeddings(text=query, model="text-embedding-ada-002")
        faq_ebd["similarity"] = faq_ebd["ada_embeddings"].apply(
            lambda x: cosine_similarity(x, query_ebd)
        )
        results = (
            faq_ebd.sort_values("similarity", ascending=False)
            .head(n)
            .completion.tolist()
        )
        for res in results:
            print(res)
            print()
