from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

claude_api_key = open('claude_2_api_key.txt', 'r').readlines()[0].strip('\n')

anthropic = Anthropic(api_key=claude_api_key)

