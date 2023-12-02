from langchain.chains.conversation.memory import ConversationBufferMemory

from aware_narratives.schemas import DEF_NAME


class DialogMemory(object):
    def __init__(self, name, self_description, long_term_goals):
        self.input_key = "input"
        self.output_key = "text"
        self.hypotheses = ""
        self.name = name
        self.self_description = self_description
        self.long_term_goals = long_term_goals
        self.short_term_goals = ""
        self.person_name = ""
        self.person_data = None
        self.examples = ""
        self.expected_role = ""

    def initialize_conversation(self, person_name):
        self.person_name = person_name
        self.buffer_memory = ConversationBufferMemory(
            human_prefix=person_name,
            ai_prefix=self.name,
            input_key=self.input_key,
            output_key=self.output_key,
        )

    def update_examples(self, examples):
        self.examples = examples

    def update_learning_goals(self, stage_goals, examples):
        self.short_term_goals = stage_goals
        self.examples = examples

    def update_hypotheses(self, hypotheses):
        self.hypotheses = hypotheses

    def update_person_data(self, data):
        self.person_data = data

    def update_expected_role(self, role):
        self.expected_role = role

    def retrieve_examples(self):
        return self.examples

    def retrieve_hypotheses(self):
        return self.hypotheses

    def retrieve_person_data(self):
        return self.person_data

    def retrieve_self_description(self):
        return self.self_description

    def retrieve_short_term_goals(self):
        return self.short_term_goals

    def retrieve_long_term_goals(self):
        return self.long_term_goals

    def retrieve_name(self):
        return self.name

    def retrieve_person_name(self):
        return self.person_name

    def retrieve_chat_history(self):
        return self.buffer_memory.buffer

    def retrieve_expected_role(self):
        return self.expected_role

    def store_conversation_turn(self, human_msg, ai_msg):
        input = {self.input_key: human_msg}
        output = {self.output_key: ai_msg}
        self.buffer_memory.save_context(inputs=input, outputs=output)

    def clear(self):
        self.buffer_memory.clear()
