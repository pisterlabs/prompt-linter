import subprocess
from pathlib import Path

from langchain.tools import BaseTool, StructuredTool, tool


def ability_runner_factory(root_path: str):
    @tool
    def run_ability(
        ability_id: str,
        ability_args: str,
    ) -> str:
        """
        Run the ability with the given id.

        :param ability_id: id of the ability to run
        :param ability_args: arguments to pass to the ability
        :return: result of the ability
        """

        cmd = "&& ".join(
            [
                f"cd '{root_path}'",
                f"source ../../start/run_ability.sh {ability_id} {ability_args}",
            ]
        )

        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Print the return code (0 is success)
        print("Return code:", result.returncode)

        # Print the output of the command
        print("Output:", result.stdout)

        # Print the stderr if any error happened
        if result.stderr:
            print("Error:", result.stderr)

        return result.stdout

    return run_ability
