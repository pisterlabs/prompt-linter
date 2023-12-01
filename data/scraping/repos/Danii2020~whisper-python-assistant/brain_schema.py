from langchain.output_parsers import ResponseSchema

title = ResponseSchema(
    name="title",
    description="Titulo del texto"
)

content = ResponseSchema(
    name="content",
    description="Contenido de los puntos clave \
generado por el LLM, por ejemplo \
1. Punto clave\n2.Punto clave."
)

response_schemas = [title, content]
