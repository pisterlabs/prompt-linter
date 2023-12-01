from utils import Utils
import openai

class _chatgpt:

  def __init__(self) -> None:
    keys = Utils.read_json("keys.json")
    openai.api_key = keys["OPENAI_API_KEY"]
    restaurants = Utils.read_json("Configs\\restaurants.json")
    file_name = f"Response\\{restaurants['NAME']}.{Utils.get_today_date()}-scrapper.json"
    reviews = Utils.read_json(file_name)
    reviews = {x['review_id']:x['review_text'] for x in reviews}

    prompt_to_chatgpt = [
      {
        "role": "system",
        "content": Utils.read_prompt_file()
      }
    ]
    for k,v in reviews.items():
      prompt_to_chatgpt.append(
        {      
          "role": "user",
          "content": k + " - " + v
        }
      )

    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt_to_chatgpt,
        temperature=0.02,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
      print(f"Call to ChatGPT sucessfully done for {restaurants['NAME']}")
      file_name = f"Response\\{restaurants['NAME']}.{Utils.get_today_date()}-chatgpt-full"
      Utils.save_response_as_json(file_name,response)
      file_name = f"Response\\{restaurants['NAME']}.{Utils.get_today_date()}-chatgpt-compact"
      Utils.save_response_as_json(file_name,response["choices"][0]["message"]["content"])
      print('The 2 files were saved')
    except Exception:
      print(f"The call to ChatGPT for {restaurants['NAME']} returned the following error {Exception}")
