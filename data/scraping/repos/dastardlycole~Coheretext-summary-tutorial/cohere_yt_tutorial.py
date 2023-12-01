import cohere
co = cohere.Client('ylDrHB4CCTOEtMqAh0zp0Dtiuvpn4Z8n9wtdngTX')
response = co.generate(
    model='xlarge',
    prompt="""Summarize this dialogue:
    Customer: Please connect me with a support agent.
    AI: Hi there, how can I assist you today?
    Customer: I forgot my password and I lost access to the email affiliated with my account.
    AI: Yes of course. First, I\'ll need to confirm your identity and then I can connect you with an agent.
    TLDR: A customer lost access to their account.
    --
    Summarize this dialogue:
    AI: Hi there, how can I assist you todayt?
    Customer: I want ot book a product demo.
    AI: Sounds greeat! What country are you located in?
    Customer: I\'m in the United States.
    TLDR: A customer wants to book a product demo.
    --
    Summarize this dialogue:
    AI: Hi there, how can I assist you today?
    Customer: I want to get more information about your pricing.
    AI: I can pull this for you, just a moment.
    TLDR:""",
    max_tokens=20,
    temperature=0.56,
    k=0,
    p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=["--"],
)
print('Prediction: {}'.format(response.generations[0].text))