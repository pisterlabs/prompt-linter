import tweepy
import time
import os
import openai


# Load your API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Helper function to generate a response for mentions


def generate_response(username):
    response_text = f"@{username} Your response here"
    return response_text


def load_last_mention_id():
    # Implement this function to load the last_mention_id from a file or database
    return None  # Replace None with the actual last_mention_id loaded from storage


def save_last_mention_id(mention_id):
    # Implement this function to save the last_mention_id to a file or database
    pass

# Main function to respond to mentions


def respond_to_mentions():
    try:
        # Get the last_mention_id to avoid duplicates
        last_mention_id = load_last_mention_id()

        # Fetch mentions since the last checked mention
        mentions = api.mentions_timeline(
            since_id=last_mention_id, tweet_mode="extended")

        for mention in reversed(mentions):
            # Respond to each mention with a dynamic response
            response_text = generate_response(mention.user.screen_name)
            api.update_status(
                status=response_text,
                in_reply_to_status_id=mention.id,
                auto_populate_reply_metadata=True
            )

            # Update the last_mention_id to avoid responding to the same mention again
            last_mention_id = mention.id

            # Save the last_mention_id to avoid duplicates
            save_last_mention_id(last_mention_id)

    except tweepy.TweepError as e:
        print("Error: ", e)


if __name__ == "__main__":
    print("Twitter bot is now running.")

    while True:
        try:
            # Call the function to respond to mentions
            respond_to_mentions()

        except Exception as e:
            # Implement comprehensive error handling and logging here
            print("Error: ", e)

        # Wait for 15 minutes before checking for new mentions again
        time.sleep(15 * 60)
