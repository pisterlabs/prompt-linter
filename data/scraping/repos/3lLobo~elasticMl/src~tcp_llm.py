import os
import socket as sc
import asyncio
from typing import List, Tuple, Generator as Yield
from pprint import pprint
import guidance as gd
try:
    from src.prompt_helpers import base_prompt, precode_prompt, postcode_prompt
except:
    from prompt_helpers import base_prompt, precode_prompt, postcode_prompt
    pass


# Create a TCP/IP socket to listen for prompts
async def listen_for_prompt(port: int = 10000) -> Yield:
    """
    Listen for prompts for the LLM on a TCP/IP socket.
    Convert the bytes to a string and yield it.
    Async function which runs parallel to the LLM main loop.
    
    Args:
        port (int): The port to listen on.
    
    Yields:
        str: The prompt.
    """
    # Create a TCP/IP socket
    sock = sc.socket(sc.AF_INET, sc.SOCK_STREAM)
    sock.bind(('localhost', port))
    # Listen for incoming connections
    sock.listen(1)

    while True:
        asyncio.sleep(0.1)
        connection, client_address = sock.accept()
        print('Established connection with', client_address)
        try:
            while True:
                prompt = ""
                is_receiving = True
                while is_receiving:
                    data = connection.recv(1024)
                    decoded_data = data.decode('utf-8')
                    if len(decoded_data) > 0:
                        prompt += decoded_data
                        if '\n' in decoded_data:
                            is_receiving = False
                    else:
                        is_receiving = False
                yield prompt
        finally:
        # Clean up the connection
            connection.close()


class StandbyLLM ():
    """
    A class to run a LLM in standby mode.
    The LLM is listening on a TCP/IP socket for prompts.
    When a prompt is received, the LLM generates a completion and sends it back.
    """
    def __init__(self, llm, port: int = 10000):
        """
        Args:
            llm (HuggingFacePipeline): The LLM.
            port (int): The port to listen on.
        """
        self.llm = llm
        self.port = port

    def run_convo(self, prompt: str) -> Yield:
        """
        Run the LLM in standby mode.
        Use guidance to enrich the prompt.

        Args:
            prompt (str): The prompt.

        Yields:
            str: The response.
        """
        precode = gd(base_prompt + precode_prompt, llm=self.llm)
        for result in precode(conversation=prompt, silent=False, stream=True):
                resolved_content = result.get('thoughts') or ''
                resolved_content += '\n```python\n'+(result.get('code') or '')+'\n```'
                # resolved_convo = [{'role': 'assistant', 'content': resolved_content}]
                yield resolved_content


if __name__ == "__main__":

    from pprint import pprint

    async def main():
        prompt_gen = listen_for_prompt(port=10001)
        first_prompt = await prompt_gen.__anext__()
        pprint(first_prompt)    

    asyncio.run(main())