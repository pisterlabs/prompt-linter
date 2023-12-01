import os
import sys
import logging
import openai

from tool import PublicodeAgent
from LlamaIndexFormatter import LlamaIndexFormatter

if os.getenv("OPENAI_URL"):
    openai.api_base = os.getenv("OPENAI_URL")
    openai.verify_ssl_certs = False

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger()

handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(LlamaIndexFormatter())
logger.addHandler(handler)

cases = [
    {
        "inputs": [
            "Calcules moi mon préavis de retraite",
            "IDCC1979",  # contrat salarié . convention collective
            "88",  # contrat salarié . ancienneté
            "oui",  # contrat salarié . mise à la retraite
            "cadre supérieure",  # contrat salarié .  convention collective . hotels cafes restaurants . catégorie professionnelle
            "non",  # contrat salarié .travailleur handicapé
        ],
        "expectations": ["91 jours"],
    },
    {
        "inputs": [
            "Calcules moi mon préavis de retraite",
            "IDCC1979",  # contrat salarié . convention collective
            "88",  # contrat salarié . ancienneté
            "oui",  # contrat salarié . mise à la retraite
            "secretaire",  # contrat salarié .  convention collective . hotels cafes restaurants . catégorie professionnelle
            "non",  # contrat salarié .travailleur handicapé
        ],
        "expectations": ["60 jours"],
    },
    {
        "inputs": [
            "Calcules moi mon préavis de retraite",
            "IDCC1043",  # contrat salarié . convention collective
            "oui",  # contrat salarié . mise à la retraite
            "88",  # contrat salarié . ancienneté
            "A",  # contrat salarié .  convention collective . gardien concierge . catégorie professionnelle
            "non",  # contrat salarié .travailleur handicapé
        ],
        "expectations": ["60 jours"],
    },
]


for case in cases:
    agent = PublicodeAgent()
    inputs = case.get("inputs", [])
    response = None
    for input_data in inputs:
        response = agent.chat(input_data)
    assert response is not None
    for expectation in case.get("expectations", []):
        assert expectation in str(
            response
        ), f"'{expectation}' not found in '{response}'"
