from extractor.config import openai
from pydantic import BaseModel
from extractor.tutorials.files import get_nf1_xml


class NfeCampos(BaseModel):
    razao_social_prestador: str
    cnpj_ou_cpf_do_prestador: str
    telefone_do_prestador: str
    email_do_prestador: str
    razao_social_tomador: str
    cnpj_ou_cpf_do_tomador: str
    valor_liquido: float
    discriminacao_dos_servicos: str


model = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=NfeCampos,
    max_retries=2,
    messages=[
        {
            "role": "user",
            "content": (
                "Você é um leitor de arquivos .xml de notas fiscais brasileiras. "
                "Sua função é somente retornar os valores respectivos das chaves especificadas."
                f"Segue o arquivo .xml: \n{get_nf1_xml()}"
            ),
        },
    ],
)


assert model.razao_social_prestador == "MUSTHOST LTDA"
assert model.cnpj_ou_cpf_do_prestador == "12884720000146"
assert model.telefone_do_prestador == "(85)3095-2008"
assert model.email_do_prestador == "vendas@musthost.com.br"
assert model.razao_social_tomador == "ASQ Asset Management LTDA"
assert model.cnpj_ou_cpf_do_tomador == ""
assert model.valor_liquido == 1790.0
