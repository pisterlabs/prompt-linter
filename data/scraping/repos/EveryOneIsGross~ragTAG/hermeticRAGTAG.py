import json
import numpy as np
from textblob import TextBlob
import random
import os
import openai
from dotenv import load_dotenv
from rake_nltk import Rake
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nltk.download('punkt')
load_dotenv()

class Agent:
    HERMETIC_PRINCIPLES = ["Mentalism", "Correspondence", "Vibration", "Polarity", "Rhythm", "Cause and Effect", "Gender"]

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.hermetic_principle = random.choice(self.HERMETIC_PRINCIPLES)
        self.memory = self.load_memory()
        ideology_data = self.load_ideology()
        self.character_description = ideology_data["character_description"]
        self.objectives = ideology_data["objectives"]
        self.temperature = 0.8
        self.ideology = f"{self.character_description} {self.objectives}"

    def load_memory(self):
        filename = self.name.replace(" ", "_") + '_memory.json'
        try:
            with open(filename, 'r') as fp:
                data = json.load(fp)
            return data.get("memory", {})
        except FileNotFoundError:
            return {}

    def save_memory(self):
        filename = self.name.replace(" ", "_") + '_memory.json'
        data = {
            "memory": self.memory
        }
        with open(filename, 'w') as fp:
            json.dump(data, fp)


    def call_ai_api(self, prompt, max_tokens=500, temperature=1, presence_penalty=0.2, frequency_penalty=1.0, n=3, echo=False, stream=False):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n,
            echo=echo,
            stream=stream
        )
        return [choice.text.strip() for choice in response.choices]

   
    def generate_ideology(self): 
        system_instruction = f"Acting strictly in character as {self.name} and guided by the Hermetic Principle of {self.hermetic_principle}, describe how your defining attributes and your unique perspective guide your decisions."
        character_description = self.call_ai_api(system_instruction, max_tokens=200, temperature=1, presence_penalty=1.5, frequency_penalty=1.2)
        
        # Generate objectives list
        system_instruction = f"Based on the Hermetic Principle of {self.hermetic_principle}, generate a list of objectives for {self.name}."
        objectives = self.call_ai_api(system_instruction, max_tokens=200, temperature=1, presence_penalty=1.5, frequency_penalty=1.2)

        persona = {
            "character_description": character_description,
            "objectives": objectives
        }

        # Save persona along with context in the memory file
        self.memory["persona"] = persona
        self.save_memory()

        # Save persona in a separate file
        self.save_ideology(persona)

        return persona
    
    def save_ideology(self, persona):
        filename = self.name.replace(" ", "_") + '_persona.json'
        with open(filename, 'w') as fp:
            json.dump(persona, fp)


    def load_ideology(self):
        filename = self.name.replace(" ", "_") + '_persona.json'
        try:
            with open(filename, 'r') as fp:
                data = json.load(fp)
            return {
                "character_description": data.get("character_description", ""),
                "objectives": data.get("objectives", "")
            }
        except FileNotFoundError:
            ideology = self.generate_ideology()
            return {
                "character_description": ideology["character_description"],
                "objectives": ideology["objectives"]
            }


    def construct_chatroom_prompt(self, agents):
        other_agents = [agent.name for agent in agents if agent.name != self.name]
        return f"You are in a chatroom with {', '.join(other_agents)}."


    def respond_to_prompt(self, user_prompt, previous_responses):
        recent_conversation = ". ".join(previous_responses[-1:])

        # Add the recent conversation to the user prompt
        user_prompt = recent_conversation + ". " + user_prompt

        # Compute the user_prompt_embedding
        user_prompt_embedding = self.get_sentence_embedding(user_prompt)

        sentiment_of_question = self.analyze_sentiment(user_prompt)

        # Retrieve previous discussions from memory if available
        past_discussions = ""
        if self.memory:
            sorted_similarities = self.check_past_answers_for_similarity(user_prompt)
            if sorted_similarities:
                top_similar_prompt, similarity = list(sorted_similarities.items())[0] # get the most similar past discussion
                if similarity > 0.7: # if similarity is high enough
                    past_discussions = f"Question: {top_similar_prompt}. Response: {self.memory[top_similar_prompt]['agent_response']}"

        # Check past answers for similarity
        similar_past_answers = self.check_past_answers_for_similarity(user_prompt)

        system_instruction = f"Considering your '{self.ideology}', your attitude towards the subject '{sentiment_of_question}', your past discussions '{past_discussions}', and similar past answers '{similar_past_answers}', answer the following question from your unique perspective as {self.name}. Do not create lists or respond with a question."
        prompt_user = f"{system_instruction} Consider the question: '{user_prompt}'"
        response_user = self.call_ai_api(prompt_user, max_tokens=250, temperature=self.temperature)

        # If response_user is a list, join it into a single string
        if isinstance(response_user, list):
            response_user = ' '.join(response_user)

        sentiment_of_response = self.analyze_sentiment(response_user)
        self.adjust_temperature(sentiment_of_response)

        # Generate multiple responses
        potential_responses = self.call_ai_api(prompt_user, max_tokens=250, temperature=self.temperature, presence_penalty=1, frequency_penalty=0.8)

        # Get sentence embeddings for each potential response
        potential_response_embeddings = [self.get_sentence_embedding(response) for response in potential_responses]

        # Compute similarity between user prompt and each potential response
        similarities = [cosine_similarity([np.array(user_prompt_embedding)], [np.array(response_embedding)])[0][0] for response_embedding in potential_response_embeddings]

        # Select response with highest similarity
        response_user = potential_responses[np.argmax(similarities)]
        
        # Compute the response_embedding
        response_embedding = self.get_sentence_embedding(response_user)

        # Extract the keywords from user_prompt and response_user using Rake
        rake = Rake(max_length=2)  # Limiting to bi-grams to get better keyword phrases
        rake.extract_keywords_from_text(user_prompt)
        user_keywords = rake.get_ranked_phrases()[:3]  # Limit to top 3 ranked phrases
        rake.extract_keywords_from_text(response_user)
        response_keywords = rake.get_ranked_phrases()[:3]  # Limit to top 3 ranked phrases

        # Add the keywords and response to the agent's memory
        self.memory[user_prompt] = {
            "agent_response": response_user,
            "user_keywords": user_keywords,
            "response_keywords": response_keywords,
            "user_prompt_embedding": user_prompt_embedding,
            "response_embedding": response_embedding
        }

        # Save the updated memory
        self.save_memory()
        
        # Record the conversation history
        self.record_conversation_history(user_prompt, response_user, user_keywords, response_keywords)
        
        return response_user


    def record_conversation_history(self, user_prompt, agent_response, user_keywords, response_keywords):
        conversation_file = f"{self.name.replace(' ', '_')}_conversation.json"
        try:
            with open(conversation_file, 'r') as fp:
                data = json.load(fp)
        except FileNotFoundError:
            data = {"conversation_history": []}

        conversation_item = {
            "agent_name": self.name,  # Add agent name to the conversation item
            "user_prompt": user_prompt,
            "agent_response": agent_response,
            "user_keywords": user_keywords,
            "response_keywords": response_keywords
        }

        data["conversation_history"].append(conversation_item)

        with open(conversation_file, 'w') as fp:
            json.dump(data, fp)


    def analyze_sentiment(self, text):
        # Use TextBlob for sentiment analysis
        testimonial = TextBlob(text)
        sentiment_score = testimonial.sentiment.polarity  # Returns a value between -1 and 1
        return sentiment_score

    def adjust_temperature(self, sentiment_score):
        if sentiment_score > 0.5:
            self.temperature = min(1.0, self.temperature + 0.1)
        elif sentiment_score < -0.5:
            self.temperature = max(0.0, self.temperature - 0.1)
        else:
            self.temperature = 0.8



    def get_sentence_embedding(self, text):
        # Load spaCy's English model
        nlp = spacy.load("en_core_web_lg")

        # Process the text
        doc = nlp(text)

        # Get the vector representation of the text (sentence embedding)
        sentence_embedding = doc.vector

        return sentence_embedding.tolist()
    
    def check_past_answers_for_similarity(self, user_prompt):
            # Get user prompt embedding
            user_prompt_embedding = self.get_sentence_embedding(user_prompt)

            # Calculate similarity between the user prompt and past responses
            similarity_scores = {}
            for prompt, memory in self.memory.items():
                # Check if response_embedding exists in memory
                response_embedding = memory.get("response_embedding")
                if response_embedding is not None:
                    similarity = cosine_similarity([np.array(user_prompt_embedding)], [np.array(response_embedding)])[0][0]
                    similarity_scores[prompt] = similarity

            # Sort the similarity scores in descending order
            sorted_similarities = {k: v for k, v in sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)}

            return sorted_similarities
    
    


def config_model():
    model_source = choose_model_source()

    if model_source == 'local':
        openai.api_base = os.getenv('LOCAL_API_BASE')
        openai.api_key = os.getenv('LOCAL_API_KEY')
        model = os.getenv('LOCAL_MODEL_PATH')
    else:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('OPENAI_ENGINE')

    return model

def choose_model_source():
    while True:
        source = input("Do you want to use a 'local' or 'online' model? ").lower()
        if source in ['local', 'online']:
            return source
        else:
            print("Invalid input. Please enter 'local' or 'online'.")

def create_agents(num_agents, model):
    agent_names_dict = ["a skeleton", "a witch", "a wizard", "a stoic", "the left-handed path", "a pragmatist", "a ghost", "a vampire", "a werewolf", "a zombie"]
    agents = []

    for i in range(num_agents):
        agent_name = input(f"Enter the name of agent {i+1} (leave blank for random selection): ")
        
        if not agent_name:
            agent_name = random.choice(agent_names_dict)
        
        agents.append(Agent(agent_name, model))
    return agents

def conduct_round_table_discussion():
    number_of_agents = int(input("Enter the number of agents you want in the discussion: "))
    model = config_model()
    agents = create_agents(number_of_agents, model)

    for agent in agents:
        agent.load_memory()

    previous_responses = []
    
    while True:
        user_prompt = input("Enter a discussion prompt or 'quit' to exit: ")
        if user_prompt.lower() == 'quit':
            break

        for agent in agents:
            print(f"\nAgent: {agent.name}\n")          
            chatroom_prompt = agent.construct_chatroom_prompt(agents)
            agent_response = agent.respond_to_prompt(user_prompt + ". " + chatroom_prompt, previous_responses)
            previous_responses.append(agent_response)
            print(f"\nResponse:\n {agent_response}\n")

        for agent in agents:
            agent.save_memory()

if __name__ == "__main__":
    conduct_round_table_discussion()
