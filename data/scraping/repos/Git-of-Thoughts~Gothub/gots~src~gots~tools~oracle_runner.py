import subprocess
from pathlib import Path

from langchain.tools import BaseTool, StructuredTool, tool


def oracle_runner_factory(root_path: str):
    @tool
    def run_oracle(
        oracle_id: str,
        oracle_args: str,
    ) -> str:
        """
        Run the oracle with the given id.

        :param oracle_id: id of the oracle to run
        :param oracle_args: arguments to pass to the oracle
        :return: result of the oracle
        """

        cmd = "&& ".join(
            [
                f"cd '{root_path}'",
                f"source ../../start/run_oracle.sh {oracle_id} {oracle_args}",
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

    return run_oracle
