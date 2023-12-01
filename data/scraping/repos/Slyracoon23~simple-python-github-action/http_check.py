import openai 
import sys
import os 


def query_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful bot.",
            },  
            {"role": "user", "content": prompt}, 
        ],
    )

    return response.choices[0].message.content

def check_response(prompt):
  response = query_gpt(prompt)
  
  print(response)

  if 'success' in response:
    print("Test Passed")
  else:
    print("Test Failed:", response)
    sys.exit(1)

if __name__ == "__main__":
  openai.api_key = sys.argv[1]
  prompt = sys.argv[2]
  check_response(prompt)