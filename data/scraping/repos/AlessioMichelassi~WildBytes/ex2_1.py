import json

import openai
import secretKeys


class ChatGptAPIWrapper:
    promptTokens = 0  # tokens utilizzati per il prompt 0.0015 dollari
    completionTokens = 0  # tokens utilizzati per la completion 0.002 dollari
    totalTokens = 0
    budget = 0.05  # 5 centesimi di dollaro

    models = {"gpt-4", "gpt-3.5-turbo", "babbage-002", "davinci-002",
              "text-davinci-003", "text-davinci-002", "davinci", "curie", "babbage", "ada"}
    model = "gpt-3.5-turbo"

    def __init__(self):
        openai.api_key = secretKeys.openAi
        try:
            data = self.open("data.json")
            self.deserialize(data)
            print(f"data loaded: {data}")
        except Exception as e:
            print(f"No data found. {e}")
            answer = input("Vuoi che creo un nuovo file? [Y/n]")
            if answer.lower() == "y":
                self.serialize()
            else:
                print("Exiting...")
                exit(1)

    def remainingTokens(self):
        """
        Calcola i tokens rimanenti partendo dal costo per 1000 tokens.
        :return:
        """
        # Costo per 1000 tokens
        costPer_1000_tokens = 0.002
        costPerToken = costPer_1000_tokens / 1000
        self.budget = self.budget - (costPerToken * self.totalTokens)
        remaining_tokens = float(self.budget / costPerToken)
        self.serialize()
        return f"Ti rimangono ${self.budget:.4f}. Puoi ancora utilizzare {int(remaining_tokens)} tokens."

    def getAnswer(self, question):
        """
        Ottiene una risposta da openAI. Ogni risposta costa 0.002 dollari.
        :param question:
        :return:
        """
        messages = [
            {"role": "user", "content": f"{question}"},
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=50, # limita la lunghezza della risposta
            temperature=0, # 0 = risposta più probabile, 1 = risposta più creativa
        )
        answer = response['choices'][0]['message']['content']
        self.promptTokens += int(response['usage']['prompt_tokens'])
        self.completionTokens += int(response['usage']['completion_tokens'])
        self.totalTokens += int(response['usage']['total_tokens'])
        return f"Risposta: {answer}\n{self.remainingTokens()}"

    def open(self, fileName):
        """
        Apre un file e ne ritorna il contenuto.
        :param fileName:
        :return:
        """
        with open(fileName, 'r', encoding="utf-8") as file:
            data = file.read()
        return data

    def save(self, fileName, data):
        """
        Salva un file.
        :param fileName: il nome del file
        :param data: il contenuto
        """
        with open(fileName, 'w', encoding="utf-8") as file:
            file.write(data)

    def serialize(self):
        """
       Serializza i dati in un file json.
       :return: i dati in formato json
       """
        data = {
            "promptTokens": self.promptTokens,
            "completionTokens": self.completionTokens,
            "totalTokens": self.totalTokens,
            "budget": self.budget,
        }
        jsonData = json.dumps(data)
        self.save("data.json", str(jsonData))
        return data

    def deserialize(self, data=None):
        """
       Deserializza i dati da un file json.
       :return: i dati in formato json
       """
        jsonData = json.loads(data)
        self.promptTokens = jsonData["promptTokens"]
        self.completionTokens = jsonData["completionTokens"]
        self.totalTokens = jsonData["totalTokens"]
        self.budget = jsonData["budget"]
        return jsonData


if __name__ == '__main__':
    ai = ChatGptAPIWrapper()
    ai.totalTokens = 100
    print(ai.remainingTokens())
    ai.totalTokens += 100
    print(ai.remainingTokens())
    ai.totalTokens += 100
    print(ai.remainingTokens())
    ai.totalTokens += 100
    print(ai.remainingTokens())


