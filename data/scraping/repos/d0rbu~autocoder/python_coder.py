import os
import subprocess
from warnings import warn
from pathlib import Path
from typing import Any, Type, Iterable, Set
from ordered_set import OrderedSet
from .coder import Coder
from .openai_coder import OpenAICoder
from .prompt_utils import system_user_prompt, system_prompt, assistant_prompt, user_prompt, tool, finish_tool
from ..tests.python_tests import PythonTests
from ..tests.tests import Tests


class PythonOpenAICoder(OpenAICoder):
    def __init__(
        self,
        key: str,
        organization: str,
        project_home: os.PathLike,
        python_executable: os.PathLike = Path("python"),
        **default_generate_config: dict[str, Any],
    ) -> None:
        super().__init__(
            key=key,
            organization=organization,
            project_home=project_home,
            **default_generate_config,
        )

        self.python_executable = python_executable

    @property
    def default_config(self) -> dict[str, Any]:
        config = super().default_config.copy()

        config.update({
            "model": "gpt-4-turbo",
        })

        return config

    @property
    def allowed_subcoders(self) -> Iterable[Type[Coder]]:
        return (PythonOpenAICoder,)

    def _files_to_tests(self, files: Set[os.PathLike], project_home: os.PathLike | None = None) -> Tests:
        return PythonTests(
            test_files=files,
            python_executable=self.python_executable,
            project_home=project_home,
        )

    PDM_PROJECT_FILES = {"pyproject.toml", "README.md", ".gitignore", "src", "tests", ".venv"}
    STANDARD_LIBRARIES = {"pytest", "pytest-dependency"}

    def scaffold(self, code_design: str, project_home: os.PathLike, max_scaffolding_steps: int = 10) -> Set[os.PathLike]:
        top_level_project_files = os.listdir(self.project_home)
        top_level_project_files = "\n".join(top_level_project_files)

        installed_dependencies = self._get_installed_dependencies()

        model_input = [
            system_prompt("You are an assistant that takes in a code design and sets up a python project that meets the code design. Setup can include setting up the project with pdm (which creates a virtual environment) and installing dependencies. The project may be fully or partially set up already including the pdm setup and dependencies. If there is nothing to do, finish immediately."),
            assistant_prompt("What is your code design?"),
            user_prompt(code_design),
            assistant_prompt("What does the top level of your project look like?"),
            user_prompt(f"In the directory {project_home}:\n{top_level_project_files}"),
            assistant_prompt("Where is your python executable located? Is it in a virtual environment?"),
            user_prompt(self.python_executable),
            assistant_prompt("What dependencies are installed?"),
            user_prompt("\n".join(installed_dependencies)),
            assistant_prompt("I will only run function calls now. Entering project scaffolding mode..."),
        ]

        tools = [
            tool(
                name="initialize_pdm_project",
                description="Run `pdm init` to initialize a pdm project.",
                parameters={
                    "type": "object",
                    "properties": {},
                },
            ),
            tool(
                name="pip_install",
                description="Pip install a package.",
                parameters={
                    "type": "object",
                    "properties": {
                        "package": {
                            "type": "string",
                            "description": "The package to install.",
                        },
                    },
                    "required": ["package"],
                },
            ),
            finish_tool,
        ]

        modified_files = set()
        for _ in range(max_scaffolding_steps):
            response = self.model(model_input=model_input, tools=tools, tool_choice="auto")
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            finished_scaffolding = False

            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "initialize_pdm_project":
                        subprocess.run(["pdm", "init", "-n", "--python", self.python_executable], cwd=self.project_home, check=True)
                        for file in self.PDM_PROJECT_FILES:
                            modified_files.add(os.path.join(self.project_home, file))

                        self.python_executable = os.path.join(self.project_home, ".venv", "Scripts", "python")
                        subprocess.run([self.python_executable, "-m", "pip", "install", *self.STANDARD_LIBRARIES], cwd=self.project_home, check=True)
                        
                        model_input.append(user_prompt("Initialized pdm project and activated `.venv` virtual environment."))
                    elif tool_call.function.name == "pip_install":
                        package = tool_call.function.arguments.get("package")

                        if package is None:
                            warn("No package specified for pip install.")
                            continue

                        subprocess.run([self.python_executable, "-m", "pip", "install", package], cwd=self.project_home, check=True)

                        model_input.append(user_prompt(f"Installed package `{package}` with python executable `{self.python_executable}`."))
                    elif tool_call.function.name == "finish":
                        finished_scaffolding = True
                    else:
                        warn(f"Unknown tool call: {tool_call}")

            if finished_scaffolding:
                break

        model_input.append(user_prompt("Thanks! ^.^"))  # hehe

        return modified_files

    def _get_installed_dependencies(self) -> Set[str]:
        command_output = subprocess.run([self.python_executable, "-m", "pip", "freeze"], capture_output=True, check=True)
        installed_dependencies = OrderedSet(command_output.stdout.decode("utf-8").strip().split("\n"))

        return installed_dependencies

    def design_solution(self, specification: Any) -> str:
        model_input = system_user_prompt("You are an assistant that helps generate design documents. Your requirements are: \n1. Design a system that meets the following specification.\n2. You must use Python.\n3. You must specify all frameworks used. (e.g. This will be built in Django and will utilize OpenCV).\n4. Make sure to be extremely detailed.", specification)
        response = self.model(model_input=model_input)
        return response.choices[0].message.content
