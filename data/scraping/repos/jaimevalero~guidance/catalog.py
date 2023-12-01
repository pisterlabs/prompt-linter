import guidance
import pandas as pd
import re
from loguru import logger

import re
import json

# TODO: Jaime. # Casos a probar Github. Me han desactivado el usuario / he perdido permisos

ticket="""
    Hola,
    Me gustaría dar de baja al usuario Jaime Valero (jaimevalero) de la organización de Telefónica de Github.
    Muchas gracias,"""

ticket="""
    Hola,
    Me gustaría dar de baja al usuario Jaime Valero (borja) de la organización de Telefónica de Github.
    Muchas gracias,"""


def enrich_message(message):

    # messages = """Buenas tardes.

    #     He tenido una incidencia usando GiHub. Al transferir el repositorio llamado 'autoVM' de mi perfil a la organización he perdido el acceso y he dejado de ser propietario. No encuentro la forma de arreglarlo, así que os escribo para ver si podéis ayudarme a solucionar el problema. Me gustaría volver a ser el propietario del repositorio junto con mi tutor @HECTOR CORDOBES DE LA CALLE. Es decir, que el repositorio tenga dos propietarios para evitar problemas como este en el futuro.

    #     Disculpad las molestias y gracias de antemano.
    #     Un saludo."""
    def detectar_elementos(texto, df):
        elementos_detectados = []
        
        for _, row in df.iterrows():
            elemento = row["elemento"]
            tipo = row["tipo"]
            if pd.isnull(elemento): continue
            if len(elemento) < 3:   continue
            if elemento.lower() == "epg"   : continue

            # Escapar caracteres especiales en el elemento
            elemento_escaped = re.escape(elemento)
            
            # Crear una expresión regular para buscar el elemento
            patron = r"\b" + elemento_escaped + r"\b"
            
            # Buscar coincidencias en el texto
            coincidencias = re.findall(patron, texto, flags=re.IGNORECASE)

            # Agregar el elemento y su tipo a la lista de elementos detectados si hay coincidencias
            if coincidencias:
                elementos_detectados.append({ "name" : elemento, "tipo" : tipo })
                # TODO: Jaime. Enriquecer con query a la BD y pasarle mas informacion

                #elementos_detectados.extend([ { "name" : elem, "tipo" tipo } for elem in coincidencias])
            # Iterar por los elementos detectados. Si es de tipo github, añadir la URL completa

            for elemento in elementos_detectados:
                if elemento["tipo"] == "repo_name":
                    elemento["url"] = "https://github.com/telefonica/" + elemento["name"]
                if elemento["tipo"] == "dn":
                    elemento["tipo"] = "full_name"  
                if elemento["name"] == "HECTOR CORDOBES DE LA CALLE" :
                    elemento["github_username"] = "hcordobest"
                if elemento["name"] == "PABLO GOMEZ ALVAREZ	" :
                    elemento["github_username"] = "pablogomez-a"

                    


        return elementos_detectados
    df = pd.read_csv('/home/jaimevalero/git/guidance/enriched.csv', sep=',')
    elementos_detectados = detectar_elementos(message, df)
    return elementos_detectados


guidance.llm = guidance.llms.OpenAI("text-davinci-003")

valid_jobs = [

    {   "name" : 'Echar o dar de baja usuario de la organization de github' , "params" : [ { "name" : "github_login"}] },
    {   "name" : 'Añadir usuario a la organization de github' , "params" : [ { "name" : "github_login"}] },
    {   "name" : 'Licencias de copilot. Habilitar copilot para usuario' , "params" : [ { "name" : "github_login"}] },
    {   "name" : 'Licencias de copilot. Deshabilitar copilot para usuario' , "params" : [ { "name" : "github_login"}] },
    {   "name" : 'Añadir miembro a team de la organización de github' , "params" : [ { "name" : "github_login"}] },
    {   "name" : 'Quitar o sacar miembro de un team de la organización de github' , "params" : [ { "name" : "github_login"}] },
    {   "name" : 'Cualquier otro caso distinto a los anteriores' , "params" : [ { "name" : "github_login"}] },
    ]



# Paso 1 Enriquecer el mensaje
enriched_message = enrich_message(ticket)

# Paso 2: Detectar el tipo de problema
def get_job_type(valid_jobs, ticket):
    if "name" in options[0]:
        options = [ job["name"] for job in valid_jobs]
    else:
        options = valid_jobs

    program = guidance('''
Which jobs about managing an organization in github could resolve the following issue. Please answer with a single phrase.
Sentence: {{ticket}}
Answer:{{select "answer" options=options}}''')
    out = executed_program = program(ticket=ticket, options=options)
    
    return out

#tipo = get_job_type(valid_jobs, ticket)
tipo = "Echar o dar de baja usuario de la organization de github"
# TODO: Jaime. Si el tipo es otro, pedir mas información, y volver lanzar el programa


# Paso 3: Detectar los parámetros necesarios para resolver el problema
def generate_job_arguments(ticket, valid_jobs, enriched_message, tipo):
    """ Given a ticket, a list of valid jobs, an enriched message and a job type, 
    ask the LLM to generate the arguments for the job

    Args:
        ticket (str): Ticket description, containing the problem to solve
        valid_jobs (_type_): Array of valid jobs, to extract the parameters
        enriched_message (_type_): Array of self discovered inventory objects detected in the ticket 
        tipo (_type_): Job that resolves the ticket
    """
    def extract_json_from_response(executed_program):
        """ Given a response from the model, extract the json from it """
        json_regex = r'json\n({.*?})'
        json_match = re.search(json_regex, executed_program.text, re.DOTALL)
        if not json_match:
            raise Exception("No json found")
        #TODO: Jaime. Enviar mensaje al usuario pidiendo mas informacion, y volver a ejecutar el programa, con la nueva info
        json_str = json_match.group(1)
        job_json =  json.loads(json_str.replace(", \n}","}").replace("""'""",'''"'''))
        logger.info(f"{job_json=}")
        return job_json

    params = [ valid_jobs["params"] for valid_jobs in valid_jobs if valid_jobs["name"] == tipo ][0] 
    logger.info(f"{params=}")
    program =  guidance('''
For the following ticket: {{ticket}}.
Describing and issue that can be resolved using the job : "{{job}}".
That needs following parameters :
{{#each params}} - {{this}}
{{/each}}
And the following self detected inventory objects : 
{{#each enriched_message}} - Object: Name:{{this.name}}, Type:{{this.tipo}}
{{/each}}
Return a json with the parameters and the values from the objects detected in the ticket. Respond only with the json
If you need more info to get the parameters, ask for it, politely in the same language the ticket is.
```json
{
    "job" : "{{job}}",
    {{#each params}} "{{this.name}}" : "{{gen 'this.name'  max_tokens=12}}", {{/each}}
}```''')

                    
    executed_program = program(
    ticket=ticket, job=tipo, 
    params=params, 
    enriched_message=enriched_message)

    #Extract json info from model
    logger.info(executed_program.text)
    job_json  =  extract_json_from_response(executed_program)
    return job_json


generate_job_arguments(ticket, valid_jobs, enriched_message, tipo)

#TODO: Jaime. Crear la info de ejecutar el job
# execute_job(job_json)

#TODO: Jaime. Enviar el resultado al usuario, en el mismo idioma que el ticket,pedir validacion
#TODO: Si el job falla, pedir mas informacion, y volver a ejecutar el programa

