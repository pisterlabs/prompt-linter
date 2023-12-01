import cohere

from os import environ

cohere_key = environ.get("COHERE_API_KEY", "")
print(cohere_key)
co = cohere.Client(cohere_key)

def elaborate(prompt):
    prediction = co.generate(
        model='large',
        prompt=prompt,
        return_likelihoods = 'GENERATION',
        max_tokens=50,
        temperature=0.7,
        num_generations=1,
        k=0,
        p=0.75)
    text = prediction.generations[0].text
    response = 'Prediction: {}'.format(".".join(text.split(".")[:-1])+".")
    return(response)

def summarize(prompt):
    summary = co.generate(
        model='xlarge',
        prompt=prompt,
        return_likelihoods = 'GENERATION',
        max_tokens=70,
        temperature=0.7,
        num_generations=1,
        k=0,
        p=0.75)
    text = summary.generations[0].text
    response = 'Summary: {}'.format(".".join(text.split(".")[:-1])+".")
    return(response)

# print(elaborate("NASA Astronaut Group 2 was the second group of astronauts selected by NASA; their names were announced on September 17, 1962. President Kennedy had announced Project Apollo, with the goal of putting a man on the Moon by the end of the decade, and more astronauts were required to fly the two-person Gemini spacecraft and three-person Apollo spacecraft then under development. The Mercury Seven had been selected for the simpler task of orbital flight, but the challenges of space rendezvous and lunar landing led to the selection of candidates with advanced engineering degrees (for four of the nine) as well as test pilot experience. The nine were Neil Armstrong, Frank Borman, Pete Conrad, Jim Lovell, James McDivitt, Elliot See, Tom Stafford, Ed White and John Young. Six of the nine flew to the Moon (Lovell and Young twice), and Armstrong, Conrad and Young walked on it as well. Seven of the nine were awarded the Congressional Space Medal of Honor."))
# print(summarize("NASA Astronaut Group 2 was the second group of astronauts selected by NASA; their names were announced on September 17, 1962. President Kennedy had announced Project Apollo, with the goal of putting a man on the Moon by the end of the decade, and more astronauts were required to fly the two-person Gemini spacecraft and three-person Apollo spacecraft then under development. The Mercury Seven had been selected for the simpler task of orbital flight, but the challenges of space rendezvous and lunar landing led to the selection of candidates with advanced engineering degrees (for four of the nine) as well as test pilot experience. The nine were Neil Armstrong, Frank Borman, Pete Conrad, Jim Lovell, James McDivitt, Elliot See, Tom Stafford, Ed White and John Young. Six of the nine flew to the Moon (Lovell and Young twice), and Armstrong, Conrad and Young walked on it as well. Seven of the nine were awarded the Congressional Space Medal of Honor."))
