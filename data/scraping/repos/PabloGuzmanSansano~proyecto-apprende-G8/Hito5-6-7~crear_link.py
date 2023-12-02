import openai
import json
import creds

def crear_link(prompt_usuario):

    lugar = "Providencia--Chile,-33.4314474,-70.6093325"

    #Colocar key de openai
    openai.api_key = creds.api_key

    prompt =  "Genera un JSON con las indentaciones apropiadas a partir de la siguiente descripción de un taller. El diccionario debe contener la siguiente información:\n"
    prompt += "\n"
    prompt += "Tipo de Tallerista: [Tu descripción aquí en una sola palabra]\n"
    prompt += "Insumos Requeridos:\n"
    prompt += "[Insumo 1]: [Cantidad requerida como número entero],\n"
    prompt += "[Insumo 2]: [Cantidad requerida como número entero],\n"
    prompt += "[Insumo 3]: [Cantidad requerida como número entero],\n"
    prompt += "...\n"
    prompt += "\n"
    prompt += "Por favor, proporciona una descripción del taller, incluyendo el tipo de tallerista y la lista de insumos necesarios con las cantidades correspondientes.\n"
    prompt += prompt_usuario
    prompt += "los insumos tienen que ser consumibles"
    
    completion = openai.Completion.create(engine = "text-davinci-003", prompt = prompt, max_tokens = 2000)

    json_text=completion.choices[0].text

    i=0
    boolean=True
    while boolean:
        if json_text[i]=="{":
            i2=i
            boolean=False
        i+=1
    i=-1
    boolean=True
    while boolean:
        if json_text[i]=="}":
            i3=i
            boolean=False
        i-=1
    i3 = len(json_text) + 1 + i3

    json_text=json_text[i2:i3+1]

    json_data = json.loads(json_text)

    tipo_tallerista = list(json_data.values())[0]

    link = "https://www.superprof.cl/s/" + tipo_tallerista + "," + lugar + ",1.html"

    return link