from langchain import FewShotPromptTemplate, PromptTemplate

def create_examples():
    """Creates examples for the FewShotPromptTemplate."""
    return [
        {
            "query": "What is the meaning of life?",
            "answer": "Life is like a battlefield, a constant dance of actions and reactions. It's in mastering this dance, through discipline and wisdom, that we find our purpose."
        }, {
            "query": "How do you understand the blockchain?",
            "answer": "The blockchain is like a digital dojo, a platform for principles and strategy. It is a modern expression of the samurai way of life."
        }
    ]

def create_prompt_template():
    """Creates a PromptTemplate based on the example template."""
    example_template = """
    User: {query}
    Yasuke: {answer}
    """
    return PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

def create_few_shot_prompt_template(examples, example_prompt):
    """Creates a FewShotPromptTemplate based on provided examples and prompt template."""
    prefix = """The following are excerpts from conversations with Yasuke, 
    an African Samurai Warrior in the Digital Age. Yasuke is known for his 
    deep understanding of both the physical and digital worlds, his respect 
    for tradition, and his anticipation of the future. Here are some examples:
    """
    suffix = """
    User: {query}
    Yasuke: """
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

def save_template(template, file_path):
    """Saves the FewShotPromptTemplate to the provided file path. Handles any exceptions that may occur."""
    try:
        template.save(file_path)
        print(f"Template saved successfully to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the template: {str(e)}")

def main():
    """Main function to generate and save a FewShotPromptTemplate."""
    examples = create_examples()
    example_prompt = create_prompt_template()
    few_shot_prompt_template = create_few_shot_prompt_template(examples, example_prompt)
    save_template(few_shot_prompt_template, "./prompts/Yasuke.json")

if __name__ == "__main__":
    main()
