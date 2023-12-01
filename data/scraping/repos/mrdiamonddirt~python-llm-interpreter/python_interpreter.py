import subprocess
import os
import datetime
import threading
import openai


class PythonInterpreter:
    def __init__(self):
        self.use_local_model = True
        self.output_file = "output/" + \
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".py"

    def run_python_code(self, filename):
        # Function to run Python code and handle errors
        try:
            code_ran = subprocess.run(
                ["python", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if code_ran.returncode == 0:
                print("Python code ran successfully")
                return code_ran.stdout
            else:
                print("Python code ran with errors")
                print("Error:", code_ran.stderr)
                return code_ran.stderr
        except subprocess.CalledProcessError as e:
            print("Error:", str(e))
            return str(e)
        except Exception as e:
            print("Error:", str(e))
            return str(e)

    def save_python_code(self, python_code, output_file):
        # Function to save Python code to a file
        try:
            with open(output_file, "w") as f:
                f.write(python_code)
            print("Python code saved as", output_file)
        except Exception as e:
            print("Error:", str(e))

    def get_user_input_for_code(self):
        save_python_code = input(
            "Do you want to save the Python code? (y/n): ")
        run_python_code = input("Do you want to run the Python code? (y/n): ")
        return save_python_code.lower() == "y", run_python_code.lower() == "y"


def run_interpreter(code=None):
    interpreter = PythonInterpreter()

    if code:
        interpreter.get_user_input_for_code()
        output = interpreter.run_python_code(code)
        if output:
            print("Python code output:\n", output)
            return output


if __name__ == "__main__":
    run_interpreter()
