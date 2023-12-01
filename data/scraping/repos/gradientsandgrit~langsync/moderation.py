from langchain.chains.moderation import OpenAIModerationChain


class StrictOpenAIModerationChain(OpenAIModerationChain):
    def _moderate(self, text: str, results: dict) -> str:
        categories_dict = results["categories"]

        # if any value is True, then the text is bad
        if any(categories_dict.values()):
            error_str = "Text was found that violates OpenAI's content policy."
            if self.error:
                raise ValueError(error_str)
            else:
                return error_str

        # call the original method
        return super()._moderate(text, results)
