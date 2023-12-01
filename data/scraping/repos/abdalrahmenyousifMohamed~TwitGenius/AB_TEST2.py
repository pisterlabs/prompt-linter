import pandas as pd
import openai

api_key = ''

def read_user_interests_csv(file_path):
    try:
        df = pd.read_csv(file_path).drop(['Unnamed: 0'], axis=1)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def generate_insights(user_interest, api_key):
    gpt_prompt = """
    Prompt: Extracting Insights, A/B Testing, and Performance Evaluation for a Tweet

    Task 1: Extract Headlines and Content Structures

    Tweet:
    {}

    Task 2: A/B Testing Suggestions

    Please suggest A/B tests for the tweet. You may propose variations in headlines or content structures.

    Task 3: Performance Metrics and Comparison

    Provide performance metrics for the original tweet, and compare them to the performance metrics of the A/B tests that you suggest.

    Additional Instructions:

    - Include a detailed explanation of the A/B tests you propose, along with reasons for their expected effectiveness.
    - Maintain clear and concise language in your responses.
    - Avoid using jargon or technical terms that may not be widely understood.

    Thank you!
    """
    
    try:
        combined_prompt = gpt_prompt.format(user_interest)

        openai.api_key = api_key

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=combined_prompt,
            max_tokens=500  
        )

        generated_response = response.choices[0].text
        return generated_response
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return None

# Main function
def main():
    file_path = '../data/trend_with_interest.csv'
    df = read_user_interests_csv(file_path)
    df= df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)


    if df is not None and not df.empty:
        user_interest = df['content_strategy'].iat[0]
        insights = generate_insights(user_interest, api_key)

        if insights:
            print("Generated Insights:")
            print(insights)
        else:
            print("Failed to generate insights.")
    else:
        print("No data available in the CSV file.")

if __name__ == "__main__":
    main()
