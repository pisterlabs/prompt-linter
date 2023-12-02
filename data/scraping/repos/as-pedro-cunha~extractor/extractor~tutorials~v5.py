import json
import os

from extractor.config import openai

filepath = os.path.join(os.path.dirname(__file__), "files", "nfe.xml")


def open_xml_as_txt(file_path):
    with open(file_path, "r") as f:
        return f.read()


functions = [
    {
        "name": "get_document_specification",
        "description": "get_document_specification",
        "parameters": {
            "type": "object",
            "properties": {
                "tipo_de_documento": {
                    "type": "string",
                    "description": "O tipo de documento que o usuário está tentando extrair.",
                    "enum": ["boleto", "nota fiscal", "desconhecido"],
                },
                "forma_de_pagamento": {
                    "type": "string",
                    "enum": [
                        "pix",
                        "boleto",
                        "cartão de crédito",
                        "cartão de débito",
                        "dinheiro",
                        "cheque",
                        "ted",
                        "doc",
                    ],
                    "description": "A forma de pagamento que o usuário está tentando extrair.",
                },
                "parcelas": {
                    "type": "integer",
                    "description": "O número de parcelas que o usuário está tentando extrair.",
                },
                "cnpj_do_vendedor": {
                    "type": "string",
                    "description": "O CNPJ do vendedor que o usuário está tentando extrair.",
                },
                "valor_total_liquido": {
                    "type": "float",
                    "description": "O valor total líquido que o usuário está tentando extrair.",
                },
                "email_do_vendedor": {
                    "type": "string",
                    "description": "O email do vendedor que o usuário está tentando extrair.",
                },
                "razao_social_do_vendedor": {
                    "type": "string",
                    "description": "A razão social do vendedor que o usuário está tentando extrair.",
                },
            },
            "required": [
                "tipo_de_documento",
                "cnpj_do_vendedor",
                "valor_total_líquido",
                "email_do_vendedor",
                "razao_social_do_vendedor",
            ],
        },
    },
]

file = open_xml_as_txt(filepath)


messages = [
    {
        "role": "system",
        "content": (
            "Respondo em JSON. Verifique se o documento atual é um boleto ou uma nota fiscal. "
            "Verifique também se a forma qual a forma de pagamento, também se tem parcelas."
            f"Segue o arquivo .xml: \n{file}"
        ),
    },
]


chat_response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=messages,
    response_format={"type": "json_object"},
    functions=functions,
    function_call={"name": "get_document_specification"},
)

json.loads(str(chat_response.choices[0].message.function_call.arguments))[
    "cnpj_do_vendedor"
]
