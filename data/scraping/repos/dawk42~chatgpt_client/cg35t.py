import openai
import os
import 
conversation_prompt = "You are a brilliant assistant who is an expert in cybersecurity."
conversation = [{"role": "system", "content": conversation_prompt}]
user_input = input_text.get("1.0", tk.END).strip()  # Retrieve the user input from the Text widget
if user_input != "":
        conversation.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=conversation,
            max_tokens=mtvar,
            temperature=tempvar,
            top_p=1.0,
            n=1,
            stop=None,
        )
        
        ai_response = response.choices[0].message.content
        display_text.insert(tk.END, "User: " + user_input + "\n")
        display_text.insert(tk.END, "AI: " + ai_response + "\n") 
        display_text.insert(tk.END, "====================" +"\n")
        conversation.append({"role": "assistant", "content": ai_response})
        time.sleep(1)
        input_text.delete("1.0", tk.END)  # Clear the user input Text widget
        scroll_to_bottom()