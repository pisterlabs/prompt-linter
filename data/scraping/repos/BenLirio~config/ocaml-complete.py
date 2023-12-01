import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

line = vim.current.range[0]
prompt = """(* OCaml *)
(* A function that takes an int and returns it's cube. *)
let cube : int -> int = fun x -> x*x*x
""" + line


text = openai.Completion.create(
        engine="code-davinci-001",
        prompt=prompt,
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["(*"]
        ).choices[0].text
for line in text.split('\n'):
    vim.current.range.append(line)
