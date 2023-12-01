import numpy as np
from associations import Association
from babyagi import openai_call

def find_new_associations(self, new_vector, subject, text):
    # We take in the subject and text of the new vector
    # We get chat gpt to look for 3 new topics
    # We query pinecone for the 3 new topics
    # We check with chat gpt if the new topics are related to the subject
    # If the are, we ask chat gpt to give us a description of the connection
    # Save the new association
    new_associations = []

    # Step 1: Take a new input vector or concept as an argument
    input_vector = new_vector

    # Step 2: Compare the input vector with other vectors stored in the system
    for association in self.associations:
        existing_vector1 = association.vector1
        existing_vector2 = association.vector2

        # Step 3: Identify potential associations based on some similarity metric
        sim1 = self.cosine_similarity(input_vector, existing_vector1)
        sim2 = self.cosine_similarity(input_vector, existing_vector2)

        # Check if the similarity is above the defined threshold
        if sim1 >= self.similarity_threshold or sim2 >= self.similarity_threshold:
            # Step 4: Create instances of the Association class for identified associations
            new_association = Association(
                input_vector, existing_vector1 if sim1 >= sim2 else existing_vector2,
                "New association found between input_vector and existing_vector"
            )

            # Step 5: Store the newly created associations (in this case, just appending to the list)
            new_associations.append(new_association)

    return new_associations


def cross_disciplinary_thinker(prompt: str, n: int = 3):
    # Construct the full prompt to ask for cross-disciplinary connections
    full_prompt = f"Think across disciplines and suggest {n} topics that are not directly related to the following knowledge, but may have valuable information or insights: {prompt}"
    
    # Call GPT API
    response = openai_call(
        prompt=full_prompt,
        max_tokens=100,
        n=1,
        temperature=0.5
    )

    # Parse the response to extract the suggestions
    suggestions = response.choices[0].text.strip().split('\n')[:n]
    
    return suggestions