from langchain.output_parsers import ResponseSchema

title = ResponseSchema(
    name="title",
    description="Titulo del input"
)

content = ResponseSchema(
    name="content",
    description="Contenido de la respuesta en puntos claves \
        generado por el LLM, por ejemplo \
            1. Punto uno\n2.Punto 2"
)

response_schemas = [title, content]