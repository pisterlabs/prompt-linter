# - After algorithm sorts into 1 or 2 emotions
# - Algorithm associates emotion with specific interest
# - Cohere.api names locations associated with given interest


import cohere
import clean_file_with_algorithm as algorithm


# Initialize
co = cohere.Client("MSuvC3ORXmJeWIzxj6D9vIw0QZAhfO6ibEmTlDYG")
interest = algorithm.emotion_giving_method(algorithm.TherapyTravel(),
                                           algorithm.InterestsList())

prompt = f"\n Name real locations in Toronto I should go to if I like " \
         f"{interest}"

print(prompt)

# moddel = medium or xlarge
response = co.generate(
    model='c1e4d1a2-5127-494b-8536-3d6845a4f267-ft',
    prompt=prompt,
    max_tokens=35,
    temperature=0.9,
    stop_sequences=["--"])

output = response.generations[0].text
print(output)
# add end_sequences
