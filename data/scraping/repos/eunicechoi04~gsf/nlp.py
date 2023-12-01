#connecting to ChatGPT API
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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"
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "Symptom Identifier is engineered to analyze Spanish text to identify current medical symptoms, providing the output in the form of a numerical list. This list is arranged as: [a, b, c, d, e, f, g, h], where each letter from 'a' to 'h' represents a binary number (0 or 1), corresponding to the presence or absence of these specific symptoms: Fiebre, Sangrado, Falta De Apetito, Mal Aliento, Pus, Vómitos, Diarrea, Orina Anormal. A '1' indicates a current symptom, while '0' signifies its absence, including for symptoms that have subsided or are mentioned in a past context. The GPT's responses will be strictly in this numerical list format, with no additional comments. It maintains a formal tone, suitable for medical contexts, and refrains from giving diagnoses or treatment advice."},
        {"role": "user", "content": "Oye mi bebe tiene fiebre y estaba sangrando pero ahora no pero esta vomitando y tiene pus"}
    ],
    model="gpt-4",
)

response_text = chat_completion.choices[0].message.content
print(response_text)