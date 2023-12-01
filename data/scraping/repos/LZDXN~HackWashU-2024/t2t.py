import os
from dotenv import load_dotenv
import openai

# Load .env file
load_dotenv()
# use the variable names as defined in .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def optimizeText(story, keywords):
    # Format the characters into a string - future work
    # characters_str = ', '.join([f"{name}: {desc}" for name, desc in characters.items()])
    
    # Format the keywords into a string
    keywords_str = ', '.join(keywords)
    
    # Prepare a prompt for the model
    prompt = f"Story: {story}\nKeywords: {keywords_str}\n\n Story is the previous chapter's conents. Write the next chpater's story based on the given story and using the keywords."
    
    # Query the model
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=75,
    )
    
    # Extract the text of the first completion generated by the model
    result = response.choices[0].text.strip()
    
    return result

def save_result_to_file(filename, result):
    with open(filename, 'w') as file:
        file.write(result)

def main():
    # Test the function with example inputs
    story = "Once upon a time in a magical kingdom, there lived a brave knight named Sir Lancelot. He went on many adventures, fought dragons, and saved the kingdom multiple times."
    # characters = {
    #                 "Sir Lancelot": "A brave knight of the kingdom.",
    #                 "Dragons": "Fearsome creatures that terrorize the kingdom."
    #             }
    keywords = ["knight", "dragons", "kingdom", "adventures"]

    # Test the function with example inputs and save the output to a text file
    result = optimizeText(story, keywords)

    # Save the result to a text file named "result.txt"
    save_result_to_file("t2tTestResult.txt", result)

    print(result)
if __name__ == '__main__':
    main()