import customtkinter as ctk
import openai

def chat_with_gpt(prompt):
    openai.api_key = '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'

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=4000,
        n=3,
        stop=None,
        temperature=0.7
    )

    return response.choices[0].text.strip()

def limpar_pesquisa():
    selecao_entry.delete(0, 'end')
    restricoes_entry.delete(0, 'end')
    resultado_text.configure(state='normal')
    resultado_text.delete('1.0', 'end')
    resultado_text.configure(state='disabled')

def exibir_receitas():
    produtos = selecao_entry.get()
    restricoes = restricoes_entry.get()

    prompt = "Quero 2 receitas e como fazê-las com os produtos de matéria-prima " + produtos + " e com as seguintes restrições: " + restricoes
    resposta = chat_with_gpt(prompt)

    resultado_text.configure(state='normal')
    resultado_text.delete('1.0', 'end')
    resultado_text.insert('end', resposta)
    resultado_text.configure(state='disabled')

    selecao_entry.delete(0, 'end')
    restricoes_entry.delete(0, 'end')

def sair():
    print("Opção: Sair")
    janela.destroy()

janela = ctk.CTk()
janela.geometry("500x400")

frame = ctk.CTkFrame(janela)
frame.pack(padx=10, pady=10)

produtos_label = ctk.CTkLabel(frame, text="Quais os produtos que você tem:")
produtos_label.pack()

selecao_entry = ctk.CTkEntry(frame)
selecao_entry.pack(pady=5)

restricoes_label = ctk.CTkLabel(frame, text="Descrição de Restrições:")
restricoes_label.pack()

restricoes_entry = ctk.CTkEntry(frame)
restricoes_entry.pack(pady=5)

resultado_scrollbar = ctk.CTkScrollbar(janela)
resultado_scrollbar.pack(side='right', fill='y')

resultado_text = ctk.CTkTextbox(janela, height=10, width=50, yscrollcommand=resultado_scrollbar.set)
resultado_text.pack(padx=10, pady=10, fill='both', expand=True)

resultado_scrollbar.configure(command=resultado_text.yview)

botao_exibir = ctk.CTkButton(janela, text="Exibir Receitas", command=exibir_receitas)
botao_exibir.pack(pady=10)

botao_limpar = ctk.CTkButton(janela, text="Limpar", command=limpar_pesquisa)
botao_limpar.pack(pady=10)

botao_sair = ctk.CTkButton(janela, text="Sair", command=sair)
botao_sair.pack(pady=10)

janela.mainloop()
