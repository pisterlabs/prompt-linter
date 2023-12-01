# import openai
from openaiclient.model.request.request import Request
from openaiclient.model.response import Response

"""
Class EditRequest

This is a derived class, with the Request class being the base class.
Handles sending requests to OpenAI for completion.
"""


class EditRequest(Request):
    def __init__(self, module):
        # initialize from parent class Request
        super().__init__()

        # pass API class and set it as cls
        self.module = module

        # setup self.requestDict

        # required values
        self.requestDict = {
            "model": "text-davinci-edit-001",
            "instruction": "",
            "input": "",
            "temperature": 1,
            "top_p": 1,
            "n": 1,
        }

        # required arguments
        self.requiredArgs = {
            "model",
            "instruction",
        }

        # optional arguments
        self.optionalArgs = set(self.requestDict.keys())
        self.optionalArgs -= self.requiredArgs

        # arguments that are settings
        self._settings = set(self.requestDict.keys())
        self._settings.remove("instruction")
        self._settings.remove("input")

    def getResponse(self):
        return Response(
            self.module.Edit.create(
                **self.requestDict,
            )
        )


if __name__ == "__main__":
    pass
