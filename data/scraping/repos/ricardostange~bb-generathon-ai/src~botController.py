from openaiBot import OpenAIBot
import prompts
from tagRunner import TagRunner
from usuario import User

class BotController:

    def __init__(self):
        self.bot = OpenAIBot()

    def processMenu(self, userMessage):
        self.bot.reset_messages()
        self.bot.add_system_message(prompts.menu)
        self.bot.add_user_message(userMessage)
        response = self.bot.get_response()
        return response
    
    def fixRawAnswer(self, rawAnswer, processedAnswer):
        self.bot.reset_messages()
        self.bot.add_system_message(prompts.fixMenu + "Mensagem Original: " + rawAnswer + "\nMensagem Provavelmente Incorreta: " + processedAnswer)
        response = self.bot.get_response()
        return response
    
    def processarQuery(self, user, query):
        tagRunner = TagRunner(user)
        response = self.processMenu(query)
        processedResponse = tagRunner.processMessage(response)
        print("Ai raw: " + response)
        print("Ai 1: " + str(processedResponse))
        if len(processedResponse) == 2:
            # Couldn't parse tags
            response = self.fixRawAnswer(response, processedResponse[1])
            processedResponse = tagRunner.processMessage(response)
            print(response)
            print(processedResponse)
            if len(processedResponse) == 2:
                print("Você poderia detalhar melhor o que deseja fazer?")
            print("Ai 2: " + str(processedResponse))
        return processedResponse
    

if __name__ == "__main__":
    botController = BotController()
    tagRunner = TagRunner(User("João", "123456", 1000, 5000))
    
    #response = botController.processMenu("Gostaria de transferir 100 reais para meu amigo")
    print("Olá, sou o assistente da sua conta virtual, como posso te ajudar? ")
    while(True):
        userInput = input()
        response = botController.processMenu(userInput)
        processedResponse = tagRunner.processMessage(response)
        print("Ai raw: " + response)
        print("Ai 1: " + str(processedResponse))
        if len(processedResponse) == 2:
            # Couldn't parse tags
            response = botController.fixRawAnswer(response, processedResponse[1])
            processedResponse = tagRunner.processMessage(response)
            print(response)
            print(processedResponse)
            if len(processedResponse) == 2:
                print("Você poderia detalhar melhor o que deseja fazer?")
            print("Ai 2: " + str(processedResponse))




