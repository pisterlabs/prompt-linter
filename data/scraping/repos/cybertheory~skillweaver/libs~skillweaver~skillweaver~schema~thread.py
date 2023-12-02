from langchain.chains.base import Chain
from skillweaver.schema import Thread

class Thread(Chain):

    name: str
    description: str
    step: int
    input: Thread

    def __init__(self, step, name = '', description = '', input = '') -> None:
        assert isinstance(name, str)
        assert isinstance(description,str)
        assert isinstance(input,Thread) or isinstance(input,str)

        pass

    def run(self, input) -> str:
        if isinstance(self.input,str):
            super.run(input)
        else:
            self.input.run(input)