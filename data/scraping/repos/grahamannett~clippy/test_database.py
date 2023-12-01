import json
import unittest
import os

import numpy as np
from sqlmodel import Session, select

from clippy.dm.db_utils import Database
from clippy.dm.data_manager import DataManager

from clippy.controllers.apis.cohere_controller import CohereController
from clippy.states.actions import Actions
from clippy.states.states import Task


class TestDatabase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.task_data_dir = "test_output/tasks/"
        self.database_path = "test_output/test_data.json"
        if os.path.exists(self.database_path):
            os.remove(self.database_path)

    async def test_save_task(self):
        dm = DataManager(task_data_dir=self.task_data_dir, database_path=self.database_path)

        task = Task.from_page(objective="test task", url="https://testing.localhost")
        dm.capture_task(task)

        task(Actions.Click(x=100, y=100, selector="testSelector"))
        task(Actions.Enter())
        step = await task.page_change_async(url="https://testing.localhost/2")
        task(Actions.Input(value="test input", x=100, y=100))
        enter_action = Actions.Enter()

        task(enter_action)

        dm.save()
        data = json.loads(open(self.database_path, "r").read())
        assert "task" in data

    async def test_save_api_calls(self):
        db = Database(self.database_path, save_api_calls=True)
        controller = CohereController()

        embedding = await controller.embed("embed a response")
        data = json.loads(open(self.database_path, "r").read())
        assert "embeddings" in data
        assert "generations" not in data

        await controller.generate(prompt="Generate a response")
        data = json.loads(open(self.database_path, "r").read())
        assert "generations" in data

        await controller.close()


class TestDataManager(unittest.IsolatedAsyncioTestCase):
    pass

    # def test_data(self):
