from time import time
from uuid import uuid4

from coherence import serialization


@serialization.proxy("Task")
@serialization.mappings({"created_at": "createdAt"})
class Task:
    def __init__(self, description: str) -> None:
        super().__init__()
        self.id: str = str(uuid4())[0:6]
        self.description: str = description
        self.completed: bool = False
        self.created_at: int = int(time() * 1000)

    def __hash__(self) -> int:
        return hash((self.id, self.description, self.completed, self.created_at))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Task):
            # noinspection PyTypeChecker
            t: Task = o
            return (
                self.id == t.id
                and self.description == t.description
                and self.completed == t.completed
                and self.created_at == t.created_at
            )
        return False

    def __str__(self) -> str:
        return 'Task(id="{}", description="{}", completed={}, created_at={})'.format(
            self.id, self.description, self.completed, self.created_at
        )
