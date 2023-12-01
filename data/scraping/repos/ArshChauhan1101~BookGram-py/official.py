import openai
import requests
import json

YOUR_API_KEY_CHAT = "API KEY"
YOUR_API_KEY_BOOKS = "API KEY"

def chat_with_bot(book_description):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant designed to "
                "suggest book names based on a given book description."
            ),
        },
        {
            "role": "user",
            "content": f"{book_description}",
        },
    ]

    # Send a message to the bot
    response = openai.ChatCompletion.create(
        model="mistral-7b-instruct",
        messages=messages,
        api_base="https://api.perplexity.ai",
        api_key=YOUR_API_KEY_CHAT,
    )

    # Extract book names from the bot's response
    book_names = response['choices'][0]['message']['content']

    # Save book names to a JSON file
    output_json = {"book_names": book_names}
    with open("book_names_output.json", "w") as json_file:
        json.dump(output_json, json_file, indent=2)

    # Print the bot's response
    print(f"Bot: {book_names}")
    print("Book names saved to 'book_names_output.json'.")

    return book_names

def get_books(api_key, query, max_results=10):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        'q': query,
        'key': api_key,
        'maxResults': max_results
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if 'items' in data:
        return data['items']
    else:
        return None

if __name__ == "__main__":
    print("Options:")
    print("1. Search by description")
    print("2. Search by name")

    user_choice = input("Enter your choice (1 or 2): ")

    if user_choice == "1":
        # Ask the user for a book description
        user_input = input("Write a book description: ")

        # Get book names from the Chat API
        book_names = chat_with_bot(user_input)
    elif user_choice == "2":
        # Ask the user for a book name
        user_input = input("Enter a book title or Author Name: ")

        # Use book names to search in Google Books API
        books_data = get_books(YOUR_API_KEY_BOOKS, user_input)
        
        if books_data:
            output_data = []  # List to store book data

            for book in books_data:
                volume_info = book['volumeInfo']
                title = volume_info['title']
                authors = volume_info.get('authors', ['Unknown'])
                published_date = volume_info.get('publishedDate', 'Unknown')
                book_category = volume_info.get('categories', ['Unknown'])

                # Create a dictionary for each book
                book_data = {
                    "Title": title,
                    "Authors": authors,
                    "Published Date": published_date,
                    "Category": book_category
                }

                output_data.append(book_data)

            # Save the data to a JSON file
            output_file_path = "BookList.json"
            with open(output_file_path, 'w') as json_file:
                json.dump(output_data, json_file, indent=2)

            print(f"Books data has been saved to {output_file_path}")
        else:
            print("No books found.")
    else:
        print("Invalid choice. Please enter 1 or 2.")

