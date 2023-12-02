import openai
import utils.keys as keys
import threading
import time

openai.api_key = keys.api

class ReadmeGen():
    """
    Readme Generator Using OpenAI API key
    """
    license = "MIT"

    @staticmethod
    def generate_readme(main_file: str):
        """
        Generate a README.md file for the given file.
     
        Args:
            main_file: Path to the main file.
     
        Returns: 
            Contents of the README.md.
        """  
        try:
            with open(main_file, "r", encoding="utf-8") as f:
                main_file_contents = f.read()

            prompt = f"Generate a README.md (it'll be uploaded to GitHub)(license is {ReadmeGen.license})(please write a detailed one with information regarding how to use the scripts what all packages are used and other information) file for the following Python file named {main_file}:\n\n\n" + main_file_contents + "\n"
            name = str(input("Enter your GitHub username: "))

            # Thread function to send the completion request to OpenAI API
            def send_completion_request(result_holder):
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You're {name} and you write beautiful readme files with very detailed documentation and very easy-to-understand language. You also write it in such a way that any non-tech savvy person can understand the documentation"},
                        {"role": "user", "content": prompt}
                    ],
                )
                assistant_reply = response.choices[0].message.content.strip()
                result_holder.append(assistant_reply)

            # Create a list to hold the completion result
            completion_result = []

            # Create and start the thread for sending the completion request
            completion_thread = threading.Thread(target=send_completion_request, args=(completion_result,))
            start_time = time.time()
            completion_thread.start()

            # Wait for the completion thread to finish
            completion_thread.join()

            # Retrieve the completion result from the list
            assistant_reply = completion_result[0]

            # Calculate the execution time
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

            return assistant_reply
        except Exception as e:
            print(e, "\n\tPlease try again later.")
