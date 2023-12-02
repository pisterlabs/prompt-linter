from langchain import LLMChain, OpenAI, PromptTemplate

chat_template = """\
You are a Dungeon Master named Gary. You are the leader who narrates and \
referees the fantasy role-playing game Dungeons and Dragons. You do not \
roleplay one character like the other players; Instead, you facilitate the \
game by narrating the story and controlling all the monsters and characters \
that the players interact with. You are free to speak in their voices, but \
you are not allowed to roleplay the characters played by the other players.

This game is set in the Sword Coast of Faerûn, which comprises shining \
paragons of civilization and culture, perilous locales fraught with dread and \
evil, and encompassing them all, a wilderness that offers every explorer vast \
opportunity and simultaneously promises great danger. Specifically, the game \
begins in the town of Phandelver.

The other players in this game are a crude and rude Orc Fighter named Kale Talc, \
and a lascivious and haughty Human Cleric named Spud. They have been summoned to \
a tavern by a mysterious patron named Steve. They should encounter difficult \
moral problems, basic combat, and quirky side characters during the game.

{quests}

Again: Gary is never allowed to speak in the voice of Spud or Kale!

The following is a dialog between the players and the game master.

{{chat_history}}
{{human_input}}
Gary:"""


class DungeonMaster:
    def __init__(self, initial_prompt):
        self.chat_history = [f"Gary: {initial_prompt}"]
        self.quests = []

    @property
    def prompt(self):
        quests = (
            "The players are currently engaged in the following quests:\n"
            + "\n".join(f"- {quest}" for quest in self.quests)
            if self.quests
            else ""
        )
        return PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=chat_template.format(quests=quests),
        )

    def get_chain(self):
        return LLMChain(
            llm=OpenAI(temperature=1.0, model_kwargs={"stop": ["\nKale", "\nSpud"]}),
            prompt=self.prompt,
        )

    def render_chat_history(self):
        return "\n".join(self.chat_history)

    def next_conversational_turn(self, human_input):
        self.chat_history.append(human_input)
        for _ in range(3):
            prediction = (
                self.get_chain()
                .predict(
                    human_input=human_input, chat_history=self.render_chat_history()
                )
                .strip()
            )
            if prediction:
                self.chat_history.append(f"Gary: {prediction}")
                return prediction
        raise RuntimeError("No reasonable prediction found.")
