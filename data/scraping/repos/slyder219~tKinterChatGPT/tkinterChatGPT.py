
import sys
import customtkinter as ctk
import openai

messages = []
temp = 0.6

sys.path.append(r"C:\Users\seanl\AppData\Local\Programs\Python\Python311\Lib\site-packages")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

key = os.environ.get("OPENAI_API_KEY")
openai.api_key = key
model = "gpt-3.5-turbo"



def enterButton(textbox, output):
    global messages

    output.configure(state = "normal")
    input = textbox.get("0.0", "end")
    messages.append({"role": "user", "content": input})

    response = openai.ChatCompletion.create(model = model,
                                                    messages = messages,
                                                    temperature = temp)
    textResponse = response.choices[0].message.content
    print(textResponse)
    messages.append({"role": "assistant", "content": textResponse})
   
    output.insert("end", "\n\n")
    output.insert("end", textResponse)
    textbox.delete("0.0", "end")

    output.configure(state = "disabled")
def main():

    root = ctk.CTk()
    root.geometry("1000x700")
    root.title("ChatGPT3.5Turbo")

    frame1 = ctk.CTkFrame(root, 
                          width = 1000,
                          height= 700)
    frame1.grid(row = 0,
                column = 0)
    
    textbox = ctk.CTkTextbox(frame1,
                             width = 1000)
    textbox.grid(row = 0,
                 column = 0,
                 sticky = "nsew")
    
    outputBox = ctk.CTkTextbox(frame1,
                               width = 1000,
                               height = 375)
    

    button = ctk.CTkButton(frame1,
                           width = 500,
                           height = 50,
                           border_width = 0,
                           corner_radius = 8,
                           text = "Enter",
                           command= lambda: enterButton(textbox, outputBox)) 
    button.grid(row = 1)
    
    
    outputBox.grid(row = 2,
                   column = 0,
                   pady = 20)
    outputBox.configure(state = "disabled")
    

    

    root.mainloop()

    
    
    

if __name__ == "__main__":
    main()
