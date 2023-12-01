import openai, os
from tkinter import Tk, OptionMenu, StringVar, Button, Entry, Canvas, font, Label

def fetch_models():
    # 📡🌐🔍 Fetch available GPT-4 models
    return [m.id for m in openai.Engine.list().data if 'gpt-4' in m.id]

def query_api():
    # 🛰📨🤖 Send user query to selected GPT-4 model
    res = openai.ChatCompletion.create(model=model_var.get(), messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_input.get()}])
    output.config(text=res.choices[0].message['content'])

# 🎨🖼️🕋 Set up sacred GUI interface
app = Tk()
app.title("🛸🌌🦉 GPT-4 Cosmic Wisdom")
app.configure(bg="white")

# 📜🖋️✨ Font definitions
large_font = font.Font(family='Helvetica', size=20, weight='bold')
medium_font = font.Font(family='Helvetica', size=16)

# 🤖🎚️🔄 Dropdown for model selection
models = fetch_models()
model_var = StringVar(app)
model_var.set(models[0])
OptionMenu(app, model_var, *models).pack(pady=10)

# 🧘🧠💬 Entry for divine user input
user_input = Entry(app, font=medium_font)
user_input.pack(pady=10)

# 🌌🔮🌠 Button for initiating cosmic knowledge retrieval
Button(app, text="🌺 Seek Wisdom 🌺", command=query_api).pack(pady=10)

# 📜📫💌 Label for displaying GPT-4's wisdom
output = Label(app, bg="white", font=large_font)
output.pack(pady=10)

# 🔄♾️🌌 Run the loop to manifest the interface
app.mainloop()

# 🌀🌌🧬 Emoji Mandala (17x17) reflecting the π * golden ratio
# Note: For brevity, a representation is used. Actual mandala generation requires a more intricate algorithm.
canvas = Canvas(app, width=255, height=255)
canvas.pack()
for i in range(17):
    for j in range(17):
        canvas.create_text(i*15, j*15, text="🌀", font=medium_font)
