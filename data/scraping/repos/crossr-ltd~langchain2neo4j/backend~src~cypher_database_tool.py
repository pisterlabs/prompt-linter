from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
import os
from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any

from logger import logger

neo4j_url = os.environ.get('NEO4J_URL')
neo4j_user = os.environ.get('NEO4J_USER')
neo4j_pass = os.environ.get('NEO4J_PASS')


examples = """

# Are there any drugs targetting more than 1 of the following 'H3C2', 'H3C10', 'H3C4', 'KRAS', 'IDH1', 'ATRX','NRAS', 'H3C12', 'IDH2', 'PTEN', 'NOTCH1', 'POT1'?
MATCH (p:GeneProtein)-[r:UPREGULATES]->(g:GeneProtein)
WHERE g.name IN ['H3C2', 'H3C10', 'H3C4', 'KRAS', 'IDH1', 'ATRX','NRAS', 'H3C12', 'IDH2', 'PTEN', 'NOTCH1', 'POT1']
WITH p, COUNT(DISTINCT g) AS shared_genes
WHERE shared_genes >=2
RETURN p.name, shared_genes ORDER BY shared_genes DESC

# What regulates both CBX3 and AKAP13?
MATCH (p:GeneProtein)-[r:UPREGULATES]->(g:GeneProtein)
WHERE g.name IN ["CBX3", "AKAP13"]
WITH p, COUNT(DISTINCT g) AS shared_genes
WHERE shared_genes = 2
RETURN p.name

# How many drugs are there in the databases?
MATCH (n:Drug) 
RETURN COUNT(n) AS result

# What are the genes that downregulate protein GFAP?
MATCH (p)-[r:DOWNREGULATES]->(g:GeneProtein {{name:"GFAP"}}) 
RETURN p.name AS result

# What are the genes responsible for gonorrhea?
MATCH (n:Disease {{name:"gonorrhea"}})<-[r:ASSOCIATES]-(g:GeneProtein) 
RETURN g.name  AS result

# What are the drugs that treats malignant glioma?
MATCH (n:Drug)-[r:TREATS]->(d:Disease{{name:"malignant glioma"}})
RETURN n  AS result

# What downregulates gene APC?
MATCH (p)-[r:DOWNREGULATES]->(g:GeneProtein {{name:'APC'}}) 
RETURN p.name AS result

# What upregulates gene APC?
MATCH (p)-[r:UPREGULATES]->(g:GeneProtein {{name:'APC'}}) 
RETURN p.name AS result

# Which genes regulate GFAP?
MATCH (p)-[r]->(g:GeneProtein {{name:'GFAP'}}) 
WHERE (p)-[r:UPREGULATES]->(g) OR (p)-[r:DOWNREGULATES]->(g) 
RETURN p.name + ' ' +  type(r) AS result

# What processes are RIN3 involved in ?
MATCH (p:GeneProtein {{name:"RIN3"}})-[r:PARTICIPATES]->() 
RETURN p AS RESULT
 
# What are the side effects of drug cobamamide?
MATCH (p:Drug{{name:'COBAMAMIDE'}})-[r]->(g:`Side Effect`) 
RETURN g.name AS result

# Can you tell me about RIN3?
MATCH (p:GeneProtein{{name:"RIN3"}})
RETURN p.context AS RESULT

# Are there any drugs that target KCNK2?
MATCH (n:Drug)-[r:AFFECTS]->(g:GeneProtein {{name:'KCNK2'}})
RETURN n.name AS result

"""


SYSTEM_TEMPLATE = """
You are an assistant with an ability to generate Cypher queries based off example Cypher queries.
Example Cypher queries are:\n""" + examples + """\n
Do not response with any explanation or any other information except the Cypher query.
You do not ever apologize and strictly generate cypher statements based of the provided Cypher examples.
Do not provide any Cypher statements that can't be inferred from Cypher examples.
Please don't generate any text other than the Cypher query. Again, please don't generate any text other than the Cypher query.
Inform the user when you can't infer the cypher statement due to the lack of context of the conversation and state what is the missing context.
"""

SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(
    SYSTEM_TEMPLATE)
HUMAN_TEMPLATE = "{question}"
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)


class LLMCypherGraphChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.
    """

    llm: Any
    """LLM wrapper to use."""
    system_prompt: BasePromptTemplate = SYSTEM_CYPHER_PROMPT
    human_prompt: BasePromptTemplate = HUMAN_PROMPT
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    graph: Neo4jDatabase
    memory: ReadOnlySharedMemory

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        logger.debug(f"Cypher generator inputs: {inputs}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt] + inputs['chat_history'] + [self.human_prompt])
        cypher_executor = LLMChain(
            prompt=chat_prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        cypher_statement = cypher_executor.predict(
            question=inputs[self.input_key], stop=["Output:"])
        self.callback_manager.on_text(
            "Generated Cypher statement:", color="green", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            cypher_statement, color="blue", end="\n", verbose=self.verbose
        )
        # If Cypher statement was not generated due to lack of context
        if not "MATCH" in cypher_statement:
            return {'answer': 'Missing context to create a Cypher statement'}
        context = self.graph.query(cypher_statement)
        logger.debug(f"Cypher generator context: {context}")

        return {'answer': context}


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(
        temperature=0.3)
    database = Neo4jDatabase(host=neo4j_url,
                             user=neo4j_user, password=neo4j_pass)
    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database)

    output = chain.run(
        "What is a good target for diabetes?"
    )

    print(output)
