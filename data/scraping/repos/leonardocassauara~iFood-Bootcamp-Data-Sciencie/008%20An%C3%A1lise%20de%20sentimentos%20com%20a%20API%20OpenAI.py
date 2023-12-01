import openai
import pandas as pd


class Feedback:
    def __init__(self, nota, comentario):
        self.nota = nota
        self.comentario = comentario


class FeedbackAnalyzer:
    def __init__(self, feedbacks):
        self.feedbacks = feedbacks

    def analisarSentimentos(self):
        comentariosFormatados = "\n".join([f"Nota: {elemento.nota} | {elemento.comentario}" for elemento in feedbacks])

        respostaAPI = openai.Completion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um modelo de análise de sentimentos com foco em feedbacks sobre experiências educacionais",
                },
                {
                    "role": "user",
                    "content": f"Analise os seguintes comentários, classificando-os em positivo, neutro, negativo: \n{comentariosFormatados}"
                }

            ]
        )
        return respostaAPI.choices[0].message.content

    def calcularNPS(self):
        detrators = sum(1 for feedback in self.feedbacks if feedback.nota <= 6)
        promoters = sum(1 for feedback in self.feedbacks if feedback.nota >=9)

        NPS = ((promoters - detrators) / len(self.feedbacks)) * 100
        return NPS


dados = pd.read_csv('/Dev Week/feedbacks.csv', delimiter=';')
feedbacks = dados.apply(lambda linha: Feedback(linha['nota'], linha['comentario']), axis=1)

openai.api_key = openaiApiKey

insights = analisarSentimentos(feedbacks)