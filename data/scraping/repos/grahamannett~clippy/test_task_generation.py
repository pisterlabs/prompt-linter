import unittest

from clippy.controllers.apis.cohere_controller import CohereController
from clippy.dm.task_bank import LLMTaskGenerator, TaskBankManager, Words, _process_word_bank


class TestTaskBank(unittest.TestCase):
    def test_words(self):
        word_bank_file = "src/taskgen/wordbank/city_bank"
        words = _process_word_bank(word_bank_file)
        assert len(words) != 0

        words = Words.from_file(word_bank_file)
        assert len(words) != 0
        word = words.sample()
        assert word != ""

    def test_wordbank(self):
        tbm = TaskBankManager()
        tbm.setup()
        value = tbm.wordbank.timeslot

    def test_examples(self):
        tbm = TaskBankManager()
        tbm.setup()

        self.assertTrue(len(tbm) > 5)

        task = tbm.sample()
        assert task != ""

        for i, task in enumerate(tbm):
            assert task != ""

            if i > 50:
                break


class TestLLMTaskGen(unittest.IsolatedAsyncioTestCase):
    async def test_llm_task_gen_with(self):
        llm_task_generator = LLMTaskGenerator()
        async with CohereController() as co:
            generated_task, raw_task = await llm_task_generator.sample(client=co, return_raw_task=True)
            self.assertTrue("{{" not in generated_task)
            self.assertTrue(("{{" in raw_task) and ("}}" in raw_task))
            self.assertTrue(raw_task.split("{{", 1)[0] in generated_task)

    async def test_llm_task_gen(self):
        llm_task_generator = LLMTaskGenerator("src/taskgen/llm")
        co = CohereController()
        generated_task, raw_task = await llm_task_generator.sample(client=co, return_raw_task=True)
        self.assertTrue("{{" not in generated_task)
        self.assertTrue(("{{" in raw_task) and ("}}" in raw_task))
        self.assertTrue(raw_task.split("{{", 1)[0] in generated_task)
        await co.close()


#
