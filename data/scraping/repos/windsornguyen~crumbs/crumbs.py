from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import time
import random
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import os


def main(testing=False):
    """
    To run: `nohup python crumbs.py &`
    To bring to foreground: `jobs -l`, then `fg {job #}`, where {job #} is from -l output}
    """
    START_HOUR = 11
    END_HOUR = 23
    MAX_FACTS = 10
    load_dotenv()
    client = OpenAI(api_key=os.getenv("API_KEY"))
    model = "gpt-4-1106-preview"
    topics = [
        "history",
        "science",
        "mathematics",
        "programming",
        "computer science",
        "machine learning",
        "art",
        "culture",
        "language",
        "language learning",
        "technology",
        "finance",
        "business",
        "chess",
        "poker",
        "gardening",
        "parenting",
        "relationships",
        "basketball trivia",
        "football trivia",
        "nature",
        "society and culture",
    ]

    facts = deque(maxlen=MAX_FACTS)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    if facts:
        tfidf_matrix = vectorizer.fit_transform(list(facts))
    else:
        tfidf_matrix = vectorizer.fit_transform(["placeholder non-empty text"])

    def calculate_cosine_similarity(new_vector, existing_vectors):
        similarities = cosine_similarity(new_vector, existing_vectors)
        return max(similarities[0])

    def is_fact_similar(new_fact, threshold=1 / 2):
        nonlocal tfidf_matrix
        new_fact_vector = vectorizer.transform([new_fact])
        max_similarity = calculate_cosine_similarity(new_fact_vector, tfidf_matrix)
        return max_similarity >= threshold

    def update_tfidf_matrix(new_fact):
        nonlocal tfidf_matrix
        facts.append(new_fact)
        tfidf_matrix = vectorizer.fit_transform(list(facts))

    def log_fact(fact, file_path="./log.txt"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(file_path, "a") as file:
            file.write(f"[{timestamp}] Fact: {fact}\n\n")

    def log_error(e, file_path="./error.log"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(file_path, "a") as file:
            file.write(f"[{timestamp}] {type(e).__name__}: {e}\n\n")

    def clean(fact):
        return fact.split("] Fact: ", 1)[1]

    def generate_context(topic, facts, max_facts=5):
        # Generate a summary from the last 'max_facts' facts about the topic
        context = ". ".join(
            [fact for fact in list(facts)[-max_facts:] if topic in fact]
        )
        print(context)
        return context

    def fetch_fact(topic, use_context=False):
        if use_context:
            recent_facts_summary = generate_context(topic, facts)
            prompt = f"Provide an interesting, one-sentence trivia fact that's educationally hyperpacked about {topic} for an eager and advanced Princeton undergraduate student hungry to learn more about the world. You should NOT generate any facts related to the following: {recent_facts_summary}."
        else:
            prompt = f"Provide an interesting, one-sentence trivia fact that's educationally hyperpacked about {topic} for an eager and advanced Princeton undergraduate student hungry to learn more about the world."

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ],
                temperature=1.5,
            )
            fact = response.choices[0].message.content.strip()
            return fact

        except Exception as e:
            log_error(e)
            return None

    def display_notification(message):
        formatted_message = message.replace('"', '\\"')
        apple_script = (
            f'display notification "{formatted_message}" with title "Daily Fact"'
        )
        subprocess.run(["osascript", "-e", apple_script])

    def is_within_operating_hours(hour, start=START_HOUR, end=END_HOUR):
        return start <= hour < end

    def fetch_unique_fact():
        temp_topics = topics.copy()
        use_context = False
        while temp_topics:
            topic = random.choice(temp_topics)
            fact = fetch_fact(topic, use_context)
            if fact and not is_fact_similar(fact):
                return fact
            else:
                temp_topics.remove(topic)
                use_context = True
                print(
                    f"Duplicate fact detected for topic '{topic}', trying a different topic... The fact was {fact}"
                )

    def schedule_fact(testing=False):
        iteration = 0
        max_iterations = 10  # For testing, run 10 iterations

        while True:
            if testing and iteration >= max_iterations:
                break  # Exit after 10 iterations in testing mode

            current_hour = time.localtime().tm_hour
            if is_within_operating_hours(current_hour) or testing:
                fact = fetch_unique_fact()
                if fact:
                    display_notification(fact)
                    log_fact(fact)
                    update_tfidf_matrix(fact)
                    if testing:
                        sleep_time = 6  # 6 seconds delay for testing
                        iteration += 1
                    else:
                        sleep_time = random.randint(
                            3600, 14400
                        )  # 1 to 4 hours in normal mode
                    time.sleep(sleep_time)
                else:
                    continue
            elif not testing:
                next_hour = (current_hour + 1) % 24
                wait_time = (next_hour - current_hour) * 3600
                time.sleep(wait_time)  # Sleep until the next hour

    def init_deque(file_path, max_entries=MAX_FACTS):
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Extract the last 'max_entries' facts
        last_facts = lines[-max_entries:] if len(lines) >= max_entries else lines

        # Clean and store the facts in the deque
        facts = deque(maxlen=max_entries)
        for line in last_facts:
            fact = clean(line)
            facts.append(fact)

        return facts

    schedule_fact(testing=testing)


if __name__ == "__main__":
    main()
