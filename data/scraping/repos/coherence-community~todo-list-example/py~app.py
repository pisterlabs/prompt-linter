from queue import Queue
from time import time
from typing import List, Any, cast, Optional
from uuid import uuid4

import jsonpickle
import quart
from coherence import NamedMap, Session, Filters, Processors
from coherence.event import MapListener
from coherence.filter import Filter
from coherence.processor import EntryProcessor
from coherence.serialization import proxy
from quart import Quart, request, redirect


# ----- Task ----------------------------------------------------------------


@proxy("Task")
class Task:
    """
    Simple Task type.  The @proxy is used to denote that this type
    will be serialized to Coherence.   The 'Task' argument to the decorator
    is the ID the type as is known by Coherence.
    """
    def __init__(self, description: str) -> None:
        super().__init__()
        self.id: str = str(uuid4())[0:6]
        self.description: str = description
        self.completed: bool = False
        self.createdAt: int = int(time() * 1000)

    def __hash__(self) -> int:
        return hash((self.id, self.description, self.completed, self.created_at))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Task):
            t: Task = cast(Task, o)
            return self.id == t.id and self.description == t.description and \
                self.completed == t.completed and self.created_at == t.created_at
        return False

    def __str__(self) -> str:
        msg: str = 'Task(id=\'{}\', description=\'{}\', completed={}, created_at={})'
        return msg.format(self.id, self.description, self.completed, self.createdAt)


# ----- Sse -----------------------------------------------------------------

class Sse:
    """
    Home-grown server-sent events handler.
    """

    def __init__(self) -> None:
        """
        Construct a new Sse instance.
        """
        self._queues: List[Queue] = []

    def queue(self) -> Queue:
        """
        Return a new queue to receive broadcast events.
        :return:
        """
        queue: Queue = Queue()
        self._queues.append(queue)
        return queue

    def broadcast(self, event_name: str, data: Any) -> None:
        """
        Broadcast the event data.

        :param event_name: the name of the event to broadcast
        :param data: the data of the event

        :return: None
        """
        msg: str = f'event: {event_name}\ndata: {data}\n\n'
        for i in range(len(self._queues)):
            self._queues[i].put_nowait(msg)


# ----- init ----------------------------------------------------------------

# the Quart application.  Quart was chosen over Flask due to better
# handling of asyncio which is required to use the Coherence client
# library
app: Quart = Quart(__name__,
                   static_url_path='',
                   static_folder='react-app')

# Create the Sse handler for this application
sse: Sse = Sse()

# NamedMap containing tasks keyed to the task ID
tasks: NamedMap[str, Task]

# the Session with the gRPC proxy
session: Session


@app.before_serving
async def init():
    """
    Creates the Session and NamedMap used by this application.
    Additionally, this creates a MapListener that will listen
    to changes in the tasks NamedMap and use the Sse instance to
    broadcast to the connected clients.
    :return:
    """

    # initialize the session using the default localhost:1408
    global session
    session = Session()

    # obtain a reference to the 'tasks' map.  All clients will use
    # this same map name.
    global tasks
    tasks = await session.get_map('tasks')

    # construct the MapListener that will be notified of changes
    # to the task map and will in turn broadcast those events
    # as server-sent events
    listener: MapListener = MapListener()
    listener.on_inserted(lambda event: sse.broadcast('insert', jsonpickle.dumps(event.new)))
    listener.on_updated(lambda event: sse.broadcast('update', jsonpickle.dumps(event.new)))
    listener.on_deleted(lambda event: sse.broadcast('delete', jsonpickle.dumps(event.old)))

    # register the listener
    await tasks.add_map_listener(listener)


# ----- routes --------------------------------------------------------------

@app.route('/')
def do_redirect():
    """
    Redirect to the static index.
    """
    return redirect("index.html", 302)


@app.route('/api/tasks/events', methods=['GET'])
def task_stream():
    """
    This route creates a server-sent event stream for the client.
    """
    
    def do_stream():
        """
        Yield the events as they come in on the queue.
        Block when no events to send.
        """
        queue: Queue = sse.queue()
        while True:
            yield queue.get()
            
    return quart.Response(do_stream(), mimetype='text/event-stream')


@app.route('/api/tasks', methods=['GET'])
async def get_tasks():
    """
    This route will return all tasks.
    :return:
    """
    completed: bool = bool(request.args.get('description'))
    filter: Filter = Filters.always() if not completed else Filters.equals('completed', True)

    tasks_list: List[Task] = []
    async for task in tasks.values(filter):
        tasks_list.append(task)

    return quart.Response(jsonpickle.encode(tasks_list, unpicklable=False), mimetype="application/json")


@app.route('/api/tasks', methods=['POST'])
async def create_task():
    """
    This route will create a new Task.
    """
    description: str = (await request.get_json())['description']
    task: Task = Task(description)
    await tasks.put(task.id, task)
    return "", 204


@app.route('/api/tasks/<id>', methods=['DELETE'])
async def delete_task(id: str):
    """
    This route will delete the task with the given id.

    :param id: the id of the task to delete
    """
    existing: Task = await tasks.remove(id)
    return "", 404 if existing is None else 200


@app.route('/api/tasks/<id>', methods=['PUT'])
async def update_task(id: str):
    """
    This route will update the description and/or the completed status
    of a Task.

    :param id: the id of the task to update
    """
    json = await request.get_json()
    description: Optional[str] = None if 'description' not in json else json['description']
    completed: Optional[bool] = None if 'completed' not in json else json['completed']

    processor: Optional[EntryProcessor[Any]] = None

    # if the description is being updated, create a Processor to update
    # it in-place within the cluster
    if description is not None and len(description) != 0:
        processor = Processors.update('description', description)

    # if the completion is being updated, create a Processor to update
    # it in-place within the cluster.
    # if updating both description and completion, compose the two processors
    # so that both properties will be updated atomically in one
    # network operation
    if completed is not None:
        completed_processor = Processors.update('completed', bool(completed))
        processor = completed_processor if processor is None else processor.and_then(completed_processor)

    result: bool = await tasks.invoke(id, processor)

    return "", 204 if result else 400


@app.route('/api/tasks', methods=['DELETE'])
async def delete_completed():
    """
    This task will delete all completed tasks.
    """
    tasks.invoke_all(Processors.conditional_remove(Filters.always()), filter=Filters.equals('completed', True))
    return "", 204


# ----- main ----------------------------------------------------------------


if __name__ == '__main__':
    # run the application on port 7003
    app.run(port=7003)
