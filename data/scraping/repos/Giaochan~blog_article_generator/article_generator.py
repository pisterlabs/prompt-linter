from openai_helper import OpenAIChat
from markdown_helper import write_to_markdown_file, sanitize_filename


def main():
    chatInstance = OpenAIChat()
    
    # Insert keywords separated by a comma
    keywords_input = input("Ingresa las keywords para tu blog: ")
    keywords = keywords_input.split(',')

    step_1 = f'''{{descripcion del blog}} = Blog de Literatura para principiantes (cualquier persona puede aprender filosofia si se le enseña correctamente)
                {{audiencia objetiva}} = 18-24 años
                {{palabras clave}} = {keywords}
                Paso 1: Proporcione por lo menos 10 títulos de publicaciones de blog optimizados para SEO y atractivos relacionados con la lista de palabras clave de SEO a continuación:
                {{palabras clave}}'''
    step_1_response = chatInstance.chat_with_gpt3(step_1)

    step_2 = '''
    de entre todos elije el que más creas que va a rankear y estructuralo de la siguiente manera:
    {{titulo del post}} = "titulo elegido"
    '''
    step_2_response = chatInstance.chat_with_gpt3(step_2)
    post_title = sanitize_filename(step_2_response.split('=')[1].strip())
    
    step_3 = chatInstance.remove_message(step_1_response)
    
    step_4 = '''
                {{tono de conversación}} = amigable y serio
                {{longitud deseada}} = 1500-2000 palabras
                Como redactor publicitario experimentado, genere un esquema de publicación de blog completo y optimizado para SEO para la palabra clave: {{palabras clave}}, dirigido a una audiencia de {{audiencia objetiva}} con un tono de conversación {{tono de conversación}} y una longitud deseada de {{longitud deseada}} .'''
    step_4_response = chatInstance.chat_with_gpt3(step_4)
    
    step_5 = '''Por cada punto del previo esquema, escribe 3 parrafos como mínimo
                Según el previo esquema de publicacion de blog, cree un articulo de blog inspirador sobre {{titulo del post}} para mi blog de {{descripcion del blog}}. Escríbalo en un tono {{tono de conversación}} pero mantenlo humano no lo uses en exceso. evite frases repetitivas y estructuras de oraciones no naturales . Usa la voz activa. Escribe mínimo 2000 palabras. Estrictamente requerido: Cada sección debe tener un mínimo de tres párrafos (requerido). Incluya las siguientes palabras clave: {{palabras clave}}. Optimizar los títulos y secciones para SEO. Utiliza negritas, listas enumeradas y de viñetas.
                Usa Markdown entre --- ---. Agrega referencias a imágenes para que la publicación sea más visual para el lector. use frases en el siguiente formato “> frase”. Usa emojis pero no en exceso ni en títulos'''
    step_5_response = chatInstance.chat_with_gpt3(step_5)

    chatInstance.show_chat_history()
    write_to_markdown_file(post_title, step_5_response)


if __name__ == "__main__":
    main()
