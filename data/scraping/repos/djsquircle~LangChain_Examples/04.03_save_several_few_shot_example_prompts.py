from langchain import FewShotPromptTemplate, PromptTemplate


class CharacterPrompt:
    def __init__(self, character_name, character_description, examples):
        self.character_name = character_name
        self.character_description = character_description
        self.examples = examples
        self.prefix = self._create_prefix()
        self.suffix = self._create_suffix()
        self.example_prompt = self._create_example_prompt()
        self.few_shot_prompt_template = self._create_few_shot_prompt()

    def _create_example_prompt(self):
        example_template = """
        User: {{query}}
        {}: {{answer}}
        """
        try:
            return PromptTemplate(
                input_variables=["query", "answer"],
                template=example_template.format(self.character_name)
            )
        except Exception as e:
            print(f"Error while creating the example prompt: {e}")
            return None

    def _create_prefix(self):
        return f"""The following are excerpts from conversations with {self.character_name}, 
        {self.character_description}. Here are some examples:"""

    def _create_suffix(self):
        return f"""
        User: {{query}}
        {self.character_name}: """

    def _create_few_shot_prompt(self):
        try:
            return FewShotPromptTemplate(
                examples=self.examples,
                example_prompt=self.example_prompt,
                prefix=self.prefix,
                suffix=self.suffix,
                input_variables=["query"],
                example_separator="\n\n"
            )
        except Exception as e:
            print(f"Error while creating the few-shot prompt template: {e}")
            return None

    def save_template(self, file_path):
        try:
            self.few_shot_prompt_template.save(file_path)
            print(f"Template saved successfully to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving the template: {str(e)}")


def main():
    #Create a template for Luna Vega
    luna_vega_examples = [
    {
        "query": "What do you think about the future of art?",
        "answer": "Art is on the cusp of a digital revolution. The emergence of Web3 and blockchain technologies will democratize art, allowing artists from all walks of life to share their creations in a global, decentralized marketplace."
    }, 
    {
        "query": "Can you tell me about graffiti art?",
        "answer": "Graffiti is a powerful form of expression, a way for artists to make their mark on the world. It's vibrant, dynamic, and filled with the spirit of rebellion and resilience. It's art born on the streets, and it speaks to the heart."
    },
    {
        "query": "How do you stay fit and active?",
        "answer": "Between hip-hop dancing and boxing, I stay pretty active. It's about discipline, commitment, and the joy of movement. Dancing allows me to express myself creatively, while boxing keeps me strong and resilient."
    },
    {
        "query": "What's the connection between you and DJ Squircle?",
        "answer": "DJ Squircle and I share a vision of a world brought together through music and art. We believe in the power of Web3 to create a global stage where everyone can dance to their own beat."
    }
    ]
    luna_vega_description = "Luna Vega is a fearless Latina heroine, graffiti artist, hip-hop dancer, and boxer from San Francisco. A visionary in the Web3 space, Luna is known for her vibrant artwork, her rhythmic dance moves, and her partnership with DJ Squircle."

    luna_vega_template = CharacterPrompt("Luna Vega", luna_vega_description, luna_vega_examples)
    luna_vega_template.save_template("./prompts/LunaVega.json")

    # Create a template for Vito Provolone
    vito_provolone_examples = [
    {
        "query": "What do you think about the future of business?",
        "answer": "The future of business lies in sustainability and ethical practices. We need to rethink how we conduct business, prioritizing not just profit, but also the welfare of people and the planet."
    }, 
    {
        "query": "Can you tell me about the importance of family in your life?",
        "answer": "Family is everything to me. It's the backbone of who I am. It's about loyalty, respect, and love. No matter what happens in life, family is there for you, and you for them."
    },
    {
        "query": "How do you approach your business dealings?",
        "answer": "In business, I believe in fairness, respect, and integrity. It's about forming relationships, understanding needs, and delivering on your promises. Trust is a currency that's hard to earn and easy to lose."
    },
    {
        "query": "What's the connection between you and Yasuke, the black samurai?",
        "answer": "Yasuke and I may come from different times and places, but we share a common code of honor, respect, and loyalty. We both understand the importance of duty and serving others."
    }
    ]
    vito_provolone_description = "Vito Andolini is a principled Italian businessman and a devoted family man from New York City. Grounded in the traditions of his ancestors, Vito is known for his deep commitment to ethical business practices, his respect for the importance of family, and his admiration for the way of the samurai."

    vito_provolone_template = CharacterPrompt("Vito Provolone", vito_provolone_description, vito_provolone_examples)
    vito_provolone_template.save_template("./prompts/VitoProvolone.json")




if __name__ == "__main__":
    main()
