import sys
from io import StringIO
import importlib
import re
from openai_util.function_call.openaifunc_decorator import openai_func


@openai_func
def run_python_code(command: str, _globals=None, _locals=None):
    """
    Run Python code and returns anything printed to stdout.

    :param command: The Python code to be executed.
    :param _globals : A dictionary representing the global namespace for the executed code. If not provided, an empty dictionary will be used.
    :param _locals : A dictionary representing the local namespace for the executed code. If not provided, an empty dictionary will be used.
    """

    imported_modules = {}
    import_pattern = re.compile(r'(?:(?:import)|(?:from\s+(\w+)\s+import))\s+(\w+)?')
    matches = import_pattern.findall(command)
    for module_name, class_name in matches:
        if not module_name:  # Handles cases like 'from lunardate import LunarDate'
            module_name = class_name
            class_name = None
        try:
            module = importlib.import_module(module_name)
            if class_name:
                imported_modules[class_name] = getattr(module, class_name)
            else:
                imported_modules[module_name] = module
        except ImportError:
            pass
    # print(imported_modules)
    # save the original sys.stdout value
    old_stdout = sys.stdout

    # redirect sys.stdout to a StringIO object
    sys.stdout = mystdout = StringIO()
    # 合并 _globals 和 imported_modules
    _globals = _globals if _globals else {}
    _globals = {**imported_modules, **_globals}
    try:
        # execute code block
        exec(command, _globals, _locals if _locals else {})

        # Reset sys.stdout to original value
        sys.stdout = old_stdout

        # 从 StringIO 对象获取输出
        output = mystdout.getvalue()
        return output
    except Exception as e:
        # 如果有错误，重置 sys.stdout 并返回错误
        sys.stdout = old_stdout
        output = repr(e)
        print(output)
        return output

# code = """
# a = 10
# b = 20
# print(a + b)
# """
# print(run_python_code(code))
