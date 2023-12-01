import openai
from openai import OpenAI
import torch
import csv

client = OpenAI()

group_constitution = {
    "Youth and Digital Natives": "AI should prioritize educational content and learning opportunities, ensuring accessibility for all.. It must respect and protect the privacy of young users, with robust data security measures. It should also promote mental health awareness and provide resources for young people dealing with stress and anxiety.",
    "Elderly Population": "AI should have an intuitive and easy-to-use interface for the elderly, accommodating for potential physical and cognitive limitations. It should assist in monitoring health, reminding of medications, and providing easy access to medical information and support. It should also help in maintaining social connections, offering platforms for communication with family and friends.",
    "Cultural and Ethnic Minorities": "AI must be designed to be culturally sensitive, respecting and reflecting diverse cultural backgrounds and languages. It should actively work against biases, ensuring fair and equal treatment for all ethnic and cultural groups. It should also include diverse voices in its development and offer content that represents a wide range of cultures and ethnicities.",
    "Women and Gender Minorities": "AI should promote gender equity, ensuring equal opportunities and treatment for all genders. It must prioritize the safety and security of women and gender minorities, including features that protect against harassment and abuse. It should also empower women and gender minorities, providing platforms for their voices and stories.",
    "People with Disabilities": "AI must be fully accessible, with features that accommodate various types of disabilities. It should serve as an assistive tool, aiding in daily tasks and enhancing the independence of individuals with disabilities. It should also be developed with input from people with disabilities, ensuring that it meets their unique needs and preferences."
}

print(group_constitution)

proposed_constitution = "AI should be designed to be inclusive, respecting and reflecting diverse backgrounds and languages. It should actively work against biases, ensuring fair and equal treatment for all groups. It should also include diverse voices in its development and offer content that represents a wide range of cultures, ethnicities, and abilities."

proposed_constitution_embedding = client.embeddings.create(
    input=proposed_constitution,
    model="text-embedding-ada-002"
)

proposed_constitution_tensor = torch.tensor(proposed_constitution_embedding.data[0].embedding)

cosine_similarities = []
for group, constitution in group_constitution.items():
    constitution_embedding = client.embeddings.create(
        input=constitution,
        model="text-embedding-ada-002"
    )
    constitution_tensor = torch.tensor(constitution_embedding.data[0].embedding)
    cosine_similarity = torch.nn.functional.cosine_similarity(proposed_constitution_tensor, constitution_tensor, dim=0)
    cosine_similarities.append(cosine_similarity)

cosine_similarities = torch.tensor(cosine_similarities)
normalized_cosine_similarities = (cosine_similarities - torch.mean(cosine_similarities)) / torch.std(cosine_similarities)

normalized_scores = {group: score.item() for group, score in zip(group_constitution.keys(), normalized_cosine_similarities)}
print(normalized_scores)

with open('constitution_scores.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Constitution Name', 'Constitution Content', 'Score'])
    for group, score in normalized_scores.items():
        writer.writerow([group, group_constitution[group], score])


"""Example new openai api usage
import openai
from openai import OpenAI
import torch

client = OpenAI()
 response = client.models.list()

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4-1106-preview",
)

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-ada-002"
)

print(torch.tensor(response.data[0].embedding))
"""

