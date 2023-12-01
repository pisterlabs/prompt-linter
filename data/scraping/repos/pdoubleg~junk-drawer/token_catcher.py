from openai import Embedding, ChatCompletion, Completion


class Usage:
    def __init__(self):
        self.__usage = []
        self.__old_embedding_create = Embedding.create
        Embedding.create = self.__new_embedding_create

        self.__old_completion_create = Completion.create
        Completion.create = self.__new_completion_create

        self.__old_chat_completion_create = ChatCompletion.create
        ChatCompletion.create = self.__new_chat_completion_create

    def __new_embedding_create(self, *args, **kwargs):
        result = self.__old_embedding_create(*args, **kwargs)
        self.__usage.append(
            {
                "model": result.get("model"),
                "tokens": result.get("usage"),
            }
        )
        return result

    def __new_completion_create(self, *args, **kwargs):
        result = self.__old_completion_create(*args, **kwargs)
        self.__usage.append(
            {
                "model": result.get("model"),
                "tokens": result.get("usage"),
            }
        )
        return result

    def __new_chat_completion_create(self, *args, **kwargs):
        result = self.__old_chat_completion_create(*args, **kwargs)
        self.__usage.append(
            {
                "model": result.get("model"),
                "tokens": result.get("usage"),
            }
        )
        return result

    def get(self):
        return self.__usage
    
    
    def total_tokens(self):
        """Return the total number of tokens collected so far."""
        return sum(d['tokens']['total_tokens'] for d in self.__usage)


      
 # Sample code

# usage = Usage()

# Execute LangChain / LlamaIndex calls

# print(usage.get())