import threading

langchain = None
imports_done = threading.Event()

def load_slow_imports():
    global langchain
    import langchain as lc
    langchain = lc
    imports_done.set()  # Signal that the import is done
    print("Slow imports done")