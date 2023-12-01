
from textwrap import dedent

import guidance

from toearthly.core import io, markdown

gpt4 = guidance.llms.OpenAI("gpt-4")

earthly_basics = io.relative_read("data/earthly_docs/basics.md")
earthly_reference = io.relative_read("data/earthly_docs/summary.md")
earthly_tips = io.relative_read("data/earthly_docs/tips.md")

input1 = io.relative_read("data/python_lint/files.md")
cot1 = io.relative_read("data/python_lint/bash_to_earthly/plan.md")
result1 = io.relative_read("data/python_lint/Earthfile")

def prompt(files: str, run: str, docker: str, build: str) -> str:
    program = guidance(
        dedent(
            """
    {{#system~}}
    You are creating an Earthfile from several bash and dockerfiles. I'll share Earthly
    documentation with you and then describe the conversion process.

    {{earthly_basics}}
    The tutorial is over. I will now describe the task.

    You are creating an Earthfile from the following inputs.
    *  `Files`: A Description of the file structure of the project. Use the file
    structure to determine what files need to be copied in at each stage of the docker
    multi-stage build.
    * `run.sh`: A bash file that wraps docker. It will call docker build and afterward
    run steps like docker push.
    * `build.Dockerfile`: A dockerfile with the correct base image to support the build
    steps. This should become the `base` and possibly the `deps` steps in the docker
    file.
    * `build.sh` A bash file that runs the build steps. These steps should become
    targets in the Earthfile.
    {{~/system}}
    {{#user~}}
    {input1}
    {{~/user}}
    {{#assistant~}}
    {{cot1}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce the files. Files that are needed need to be copied in.
    {{~/user}}
    {{#assistant~}}
    {{result1}}
    {{~/assistant}}
    {{#user~}}
    `Files:`
    ```
    {{files}}
    ```

    `run.sh`:
    ```
    {{run}}
    ```

    build.Dockerfile
    ```
    {{docker}}
    ```

    `build.sh`:
    ```
    {{build}}
    ```

    An Earthfile is a better way to represent this build process because it combines the
    concepts of running bash commands to build something with the ideas of
    containerisation made popular by Docker and dockerfile.

    Task:
    Do not produce the Earthfile. Instead,  describe how you would approach this
    problem. Then go through the files, step by step, and discuss how the steps should
    be ported to Earthly.
    {{~/user}}
    {{#assistant~}}
    {{gen "discuss" temperature=0 max_tokens=2000}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce the files. Files that are needed need to be copied in.
    {{~/user}}
    {{#assistant~}}
    {{gen "Earthfile" temperature=0 max_tokens=2000}}
    {{~/assistant}}
    """
        ),
        llm=gpt4,
    )
    out = io.run_llm_program(
        program,
        earthly_basics=earthly_basics,
        input1=input1,
        cot1=cot1,
        result1=result1,
        files=files,
        run=run,
        docker=docker,
        build=build,
    )
    io.write_debug("plan.md", out["discuss"], "bash_to_earthly")
    io.write_debug("result.md", out["Earthfile"], "bash_to_earthly")
    results = markdown.extract_code_blocks(out["Earthfile"])
    if len(results) != 1:
        raise ValueError(f"1 Files exepected back. Instead got {len(results)}.")
    earthfile = results[0]
    io.write_debug("Earthfile", earthfile, "bash_to_earthly")
    return earthfile
