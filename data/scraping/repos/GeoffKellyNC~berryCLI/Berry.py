import openai
from termcolor import colored
import requests


class Berry:
    def __init__(self, apiKey: str, engine: str = 'gpt-3.5-turbo') -> None:
        self._apiKey: str = apiKey
        self._engine: str = engine
        self._baseURL: str = 'https://api.openai.com/v1/'
        self._context: list[dict[str, str]] = [{"role": "system", "content": "You are a cool hip young ai."} ]
        self._tokensUsed: int = 0
        self._priceRate: dict[str, float] = {
            "gpt-3.5-turbo": 0.000002,
            "other-engine": 0.000001  
        }
        self._sessionPrice: float = 0
        self._headers: dict = {"Authorization": f"Bearer {self._apiKey}"}
        
    @property
    def _getEngine(self) -> str:
        return self._engine
    
    def _addToContext(self, response: dict) -> None:
        self._context.append(response)
        return
        
    def _getContext(self) -> list:
        return self._context
    
    def _updateUsedTokens(self, newTokens) -> None:
        self._tokensUsed = newTokens
        
    def getTokensUsed(self) -> int:
        return self._tokensUsed
    
    def addHeader(self,key: str, value: str) -> None:
        self._headers[key] = value
        return
    
    def getEngines(self): 
        try:
            engineRes = requests.get(self._baseURL + 'models', headers = self._headers)
            engineRes.raise_for_status()
            engineData = engineRes.json()
            return engineData["data"]
        except requests.exceptions.HTTPError as e:
            print(f'HTTP ERROR: {e}')
            return
        except requests.exceptions.RequestException as e:
            print(f'Error Occurred: {e}')
            return
    
    def engineInfo(self, engineId):
        try:
            openAiRes = requests.get(self._baseURL + f'models/{engineId}', headers = self._headers)
            openAiRes.raise_for_status()
            engineData = openAiRes.json()
            print(engineData)
            return
        except requests.exceptions.HTTPError as e:
            print(f'HTTP ERROR: {e}')
            return
        except requests.exceptions.RequestException as e:
            print(f'Error Occurred: {e}')
            return
    
    def increaseTokens(self, newTokens) -> int:
        newTotal: int = self._tokensUsed + newTokens
        self._updateUsedTokens(newTotal)
        return newTotal
    
    def updateSessionPrice(self, tokens) -> int:
        self._sessionPrice = self._priceRate.get(self._engine) * tokens
        return self._sessionPrice
        
    
    def askTurbo(self, prompt: str) -> None:
        openai.api_key: str = self._apiKey
        
        self._addToContext({"role": "user", "content": prompt})
        
        message: list[dict[str,str]] = self._getContext()
        
        completion: dict = openai.ChatCompletion.create(model = self._engine, messages = message)
        
        aiRes = completion.choices[0].message
        
        self._addToContext({"role": "assistant", "content": aiRes["content"]})
        
        totalTokensSession = self.increaseTokens(completion.usage["total_tokens"])
        
        sessionPrice = self.updateSessionPrice(completion.usage["total_tokens"])
        
        return {
                "response": aiRes["content"], 
                "usage": completion.usage,
                "sessionTokenTotal": totalTokensSession,
                "sessionPrice": sessionPrice
                }
        
    #? Modifiers:
    # g - General - General prompt is used
    # h - Healthy - prompt that gets the healthiest recipe
    # s - Savings - Cheapest Option
    # q - Quick - Add quick modifier.
    
    #? Modifier Variations
    # gs - General And Savings
    # hs - Healthy and Savings
    #
    
    def getRecipes(self, items: list[str], modifier: str = None):
        openai.api_key: str = self._apiKey
        
        generalPrompt: str = f"Hey GPT, I am in the mood for a recipe that includes the following ingredients: {', '.join(items)}. Can you suggest a recipe for me?"
        
        completion: dict = openai.ChatCompletion.create(model = self._engine, messages = [{"role": "user", "content": generalPrompt}])
        
        aiRes = completion.choices[0].message
        
        totalTokensSession = self.increaseTokens(completion.usage["total_tokens"])
        
        sessionPrice = self.updateSessionPrice(completion.usage["total_tokens"])
        
        return {
                "response": aiRes["content"], 
                "usage": completion.usage,
                "sessionTokenTotal": totalTokensSession,
                "sessionPrice": sessionPrice
                }
        
        
        
        
        
        
        
        
    
    
        