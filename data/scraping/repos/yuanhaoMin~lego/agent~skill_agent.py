import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma
from constant.prompt import skill_prompt
from util.ansi_console_utils import print_skill_agent_message
from util.file_utils import dump_text
from util.json_utils import dump_json, load_json

SKILL_DIR = "./skill"
SKILL_CODE_DIR = f"{SKILL_DIR}/code"
SKILL_DESCRIPTION_DIR = f"{SKILL_DIR}/description"
SKILL_VECTORDB_DIR = f"{SKILL_DIR}/vectordb"
SKILL_JSON_PATH = f"{SKILL_DIR}/skills.json"


class SkillAgent:
    def __init__(
        self,
        model_name: str,
        request_timout: int,
        resume: bool,
        retrieval_top_k: int,
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=request_timout,
        )
        os.makedirs(SKILL_CODE_DIR, exist_ok=True)
        os.makedirs(SKILL_DESCRIPTION_DIR, exist_ok=True)
        os.makedirs(SKILL_VECTORDB_DIR, exist_ok=True)
        self.retrieval_top_k = retrieval_top_k
        if resume:
            print_skill_agent_message(f"正在从检查点恢复技能代理")
            self.skills = load_json(SKILL_JSON_PATH)
        else:
            self.skills = {}
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=SKILL_VECTORDB_DIR,
        )

    def add_new_skill(self, info) -> None:
        program_call = info["program_call"]
        program_code = info["program_code"]
        program_name = info["program_name"]
        skill_description = self.generate_skill_description(
            program_code=program_code,
            program_name=program_name,
        )
        print_skill_agent_message(f"已生成技能描述:\n{skill_description}")
        if program_name in self.skills:
            print_skill_agent_message(f"技能 {program_name} 已存在，正在覆盖")
            self.vectordb._collection.delete(ids=[program_name])
            i = 2
            while f"{program_name}V{i}.py" in os.listdir(SKILL_CODE_DIR):
                i += 1
            dumped_program_name = f"{program_name}V{i}"
        else:
            dumped_program_name = program_name
        self.vectordb.add_texts(
            texts=[skill_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )
        self.skills[program_name] = {
            "code": program_code,
            "description": skill_description,
        }
        assert self.vectordb._collection.count() == len(
            self.skills
        ), "vectordb is not synced with skills.json"
        dump_text(
            text=program_code, file_path=f"{SKILL_CODE_DIR}/{dumped_program_name}.py"
        )
        dump_text(
            text=skill_description,
            file_path=f"{SKILL_DESCRIPTION_DIR}/{dumped_program_name}.txt",
        )
        dump_json(self.skills, SKILL_JSON_PATH)
        self.vectordb.persist()

    def generate_skill_description(self, program_code, program_name) -> str:
        human_message_content = (
            program_code + "\n" + f"The main function is `{program_name}`."
        )
        messages = [
            SystemMessage(content=skill_prompt.content),
            HumanMessage(content=human_message_content),
        ]
        function_description = self.llm(messages).content
        skill_description = program_name + " {\n" + function_description + "\n}"
        return skill_description

    def retrieve_skills(self, query):
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        print_skill_agent_message(f"正在检索最相似的 {k} 个技能")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        print_skill_agent_message(
            f"检索到以下技能:\n{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}"
        )
        skills = []
        for doc, _ in docs_and_scores:
            skills.append(self.skills[doc.metadata["name"]]["code"])
        return skills
