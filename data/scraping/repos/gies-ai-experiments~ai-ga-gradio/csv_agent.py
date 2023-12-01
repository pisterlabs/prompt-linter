from langchain.agents import create_csv_agent, AgentType

import utils


class CSVAgent:
    def __init__(self, llm, embeddings, file):
        self.llm = llm
        self.file = file
        self.summary_index_name = "canvas-discussions-summary"
        self.folder_path = "vector_stores/"
        self.summary_index_file = "vector_stores/canvas-discussions-summary.faiss"
        self.summary_pickle_file = "vector_stores/canvas-discussions-summary.pkl"
        self.summary_docs = utils.get_csv_files(self.file, source_column='student_name')
        self.summary_index = self.get_search_index(embeddings)
        self.agent = self.create_agent()

    def get_search_index(self, embeddings):
        if utils.index_exists(self.summary_pickle_file, self.summary_index_file):
            # Load index from pickle file
            search_index = utils.load_index(self.folder_path, self.summary_index_name, embeddings)
        else:
            search_index = utils.create_index(self.folder_path, self.summary_index_name, embeddings, self.summary_docs)
            print("Created index")
        return search_index

    def create_agent(self):

        agent = create_csv_agent(
            self.llm,
            self.file,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent
