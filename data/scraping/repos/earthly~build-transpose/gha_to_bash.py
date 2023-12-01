from textwrap import dedent
from typing import Tuple

import guidance

from toearthly.core import io, markdown

gpt4 = guidance.llms.OpenAI("gpt-4")

input1 = io.relative_read("data/python_lint/workflow.yml")
cot1 = io.relative_read("data/python_lint/gha_to_bash/plan.md")
result1 = io.relative_read("data/python_lint/gha_to_bash/result.md")

input2 = io.relative_read("data/docker_simple/workflow.yml")
cot2 = io.relative_read("data/docker_simple/gha_to_bash/plan.md")
result2 = io.relative_read("data/docker_simple/gha_to_bash/result.md")

def prompt(gha: str, files: str) -> Tuple[str, str, str]:
    program = guidance(
        dedent(
            """
    {{#system~}}
    Given a GitHub Actions workflow YAML file, summarize how you would recreate the
    steps of this build using bash and docker.

    The implementation will consist of a run.sh script that creates and runs a Docker
    container where our build.sh script is executed. This approach encapsulates our
    build process in a controlled environment (the Docker container), isolating it from
    the host machine and ensuring that it has all the necessary dependencies, regardless
    of where or on what machine the build is running. This is why we choose to use
    Docker and why we run the build process inside a Docker container, even though it
    may seem like overkill for some simple build processes.

    You will create three files:
    * `run.sh`: A bash file that wraps docker. It will call docker build and afterward
    run steps like docker push. Steps like git cloning and changing into the repo
    aren't needed because this file is stored in the git repository along with the
    code.
    * `build.Dockerfile`: A dockerfile with the correct base image to support the
    build steps. This includes any programming language tool and any dependencies needed
    for `build.sh`. If no special dependencies are needed choose alpine.
    * `build.sh` A bash file that runs the steps of the build process. It will run
    inside `build.Dockerfile` in the working directory of the repository. If no build
    steps need to run inside the container, you just include an `echo` as a placeholder.

    Other files will exist in the repository. Code files and other assets and possibly
    an application Dockerfile. The application Dockerfile you can call `app`.

    Important considerations:
    * no need to install dependencies, nor check out the code in `build.sh` because it
    is run inside the image produced from `build.Dockerfile`. Call that docker image
    `build`.
    * `build.Dockerfile` should work without volume mounting. Files that are needed need
    to be copied in.
    * References to building/tagging and pushing a Docker image or container in GitHub
    Actions workflow YAML do not refer to `build.Dockerfile` and `build` but to the
    application `app` Dockerfile called `Dockerfile`.
    * Any pushing and tagging of images should be of images made from  app `Dockerfile`
    and not from `build.Dockerfile`. Docker image `build` is used strictly for building
    steps and is used as a way to set up dependencies for that build in a repeatable
    way.
    * Don't include any steps that executing the git hub action wouldn't produce. This
    may mean a step does nothing.
    * You do not need to chmod `build.sh` or `run.sh`. That is taken care of.

    Do not produce the files. Instead, describe how you would approach this problem.
    Then go through the yaml document section by section and discuss if steps should be
    included or omitted, which of the three files it should be in, and how it needs to
    be adapted to the new format.
    {{~/system}}

    {{~! Training Example 1 ~}}
    {{#user~}}
    {input1}
    {{~/user}}
    {{#assistant~}}
    {{cot1}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce `run.sh`,`build.Dockerfile` and `build.sh`.
    Remember `build.Dockerfile` should work without volume mounting: files that are
    needed need to be copied in.
    And three files should be produced, even if they are just place holders.
    {{~/user}}
    {{#assistant~}}
    {{result1}}
    {{~/assistant}}

    {{~! Training Example 2 ~}}
    {{#user~}}
    {input2}
    {{~/user}}
    {{#assistant~}}
    {{cot2}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce `run.sh`,`build.Dockerfile` and `build.sh`.
    Remember `build.Dockerfile` should work without volume mounting: files that are
    needed need to be copied in.
    And three files should be produced, even if they are just place holders.
    {{~/user}}
    {{#assistant~}}
    {{result2}}
    {{~/assistant}}

    {{~! Generate Answer~}}
    {{#user~}}
    Files:
    ```
    {{files}}
    ```

    GitHub Actions workflow:
    ```
    {{gha}}
    ```
    {{~/user}}
    {{#assistant~}}
    {{gen "discuss" temperature=0 max_tokens=2000}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce `run.sh`,`build.Dockerfile` and `build.sh`.
    Remember `build.Dockerfile` should work without volume mounting: files that are
    needed need to be copied in.
    And three files should be produced, even if they are just place holders.
    {{~/user}}
    {{#assistant~}}
    {{gen "files" temperature=0 max_tokens=500}}
    {{~/assistant}}
    """
        ),
        llm=gpt4,
    )

    out = io.run_llm_program(
        program,
        gha=dedent(gha),
        files=files,
        input1=input1,
        cot1=cot1,
        result1=result1,
        input2=input2,
        cot2=cot2,
        result2=result2,
    )
    io.write_debug("plan.md", out["discuss"], "gha_to_bash")
    io.write_debug("result.md", out["files"], "gha_to_bash")
    results = markdown.extract_code_blocks(out["files"])
    if len(results) != 3:
        raise ValueError(f"3 Files exepected back. Instead got {len(results)}")
    io.write_debug("run.sh", results[0], "gha_to_bash")
    io.write_debug("build.Dockerfile", results[1], "gha_to_bash")
    io.write_debug("build.sh", results[2], "gha_to_bash")
    return (results[0], results[1], results[2])
