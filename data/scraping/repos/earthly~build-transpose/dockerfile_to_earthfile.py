
from textwrap import dedent

import guidance

from toearthly.core import io, markdown

gpt4 = guidance.llms.OpenAI("gpt-4")

earthly_basics = io.relative_read("data/earthly_docs/basics.md")
earthly_reference = io.relative_read("data/earthly_docs/summary.md")
earthly_tips = io.relative_read("data/earthly_docs/tips.md")

examples = [{
        "docker":   io.relative_read("data/docker_simple2/Dockerfile"),
        "workflow": io.relative_read("data/docker_simple2/workflow.yml"),
        "plan":     io.relative_read("data/docker_simple2/dockerfile_to_earthfile/plan.md"),  # noqa: E501
        "result":   io.relative_read("data/docker_simple2/dockerfile_to_earthfile/result.md")  # noqa: E501
    },
    {
        "docker":   io.relative_read("data/docker_multistage1/Dockerfile"),
        "workflow": io.relative_read("data/docker_multistage1/workflow.yml"),
        "plan":     io.relative_read("data/docker_multistage1/dockerfile_to_earthfile/plan.md"),  # noqa: E501
        "result":   io.relative_read("data/docker_multistage1/dockerfile_to_earthfile/result.md")  # noqa: E501
    },
    {
        "docker":   io.relative_read("data/docker_multistage2/Dockerfile"),
        "workflow": io.relative_read("data/docker_multistage2/workflow.yml"),
        "plan":     io.relative_read("data/docker_multistage2/dockerfile_to_earthfile/plan.md"),  # noqa: E501
        "result":   io.relative_read("data/docker_multistage2/dockerfile_to_earthfile/result.md")  # noqa: E501
    },
    ]


# Throws openai.error.InvalidRequestError
# This model's maximum context length is 8192 tokens. However, you requested 8480
# tokens (7480 in the messages, 1000 in the completion). Please reduce the length
# of the messages or completion.
# ToDo: recover from this by downgrading to GPT3.5
def prompt(docker: str, build: str) -> str:

    program = guidance(
        dedent(
            """
    {{#system~}}
    You are creating an Earthfile from a Dockerfile and a GitHub Actions workflow. I'll
    share Earthly documentation with you and then describe the conversion process.

    {{earthly_basics}}
    {{earthly_tips}}
    The tutorial is over. I will now describe the task.

    You are creating an Earthfile from the following inputs.
    * A Dockerfile: each stage in the Dockerfile will become a target in the Earthfile.
    * A GitHub Action workflow: This may not be needed. Only steps in workflow which
      describe docker actions like tagging or pushing or running docker with certain
      arguments may be relevant. The rest should be ignored.

    {{~/system}}
    {{~#each examples}}
    {{#user~}}
    Github Actions Workflow:
    ```
    {{this.workflow}}
    ```

    Dockerfile:
    ```Dockerfile
    {{this.docker}}
    ```

    Task:
    Do not produce the Earthfile. Instead, describe how you would approach this
    problem. Then go through the files, step by step, and discuss how the steps should
    be ported to an Earthfile.

    Remember:
    - an Earthfile can't have a target named base.
    - an Earthfile `COPY` from another target works like a Dockerfile multistage COPY
       but it has a different syntax.
    - To copy `example` from target `+build` use `COPY +build/example .`
    - Also, `example` will need to be saved using `SAVE ARTIFACT` in `+build`

    Let me go step by step through the dockerfile and convert it to a Earthfile.
    {{~/user}}
    {{#assistant~}}
    {{this.plan}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce the Earthfile in backticks.
    {{~/user}}
    {{#assistant~}}
    {{this.result}}
    {{~/assistant}}
    {{~/each}}
    {{#user~}}
    Github Actions Workflow:
    ```
    {{build}}
    ```

    Dockerfile:
    ```Dockerfile
    {{docker}}
    ```

    Task:
    Do not produce the Earthfile. Instead, describe how you would approach this
    problem. Then go through the files, step by step, and discuss how the steps should
    be ported to an Earthfile.

    Remember:
    - an Earthfile can't have a target named base.
    - an Earthfile `COPY` from another target works like a Dockerfile multistage COPY
       but it has a different syntax.
    - To copy `example` from target `+build` use `COPY +build/example .`
    - Also, `example` will need to be saved using `SAVE ARTIFACT` in `+build`

    Let me go step by step through the dockerfile and convert it to a Earthfile.
    {{~/user}}
    {{#assistant~}}
    {{gen "discuss" temperature=0 max_tokens=1000}}
    {{~/assistant}}
    {{#user~}}
    Ok, produce the Earthfile in backticks.
    {{~/user}}
    {{#assistant~}}
    {{gen "Earthfile" temperature=0 max_tokens=500}}
    {{~/assistant}}
    """
        ),
        llm=gpt4,
    )
    out = io.run_llm_program(
        program,
        earthly_basics=earthly_basics,
        earthly_reference=earthly_reference,
        earthly_tips=earthly_tips,
        examples=examples,
        docker=docker,
        build=build,
    )
    io.write_debug("plan.md", out["discuss"], "dockerfile_to_earthfile")
    io.write_debug("result.md", out["Earthfile"], "dockerfile_to_earthfile")
    results = markdown.extract_code_blocks(out["Earthfile"])
    if len(results) != 1:
        raise ValueError(f"1 Files exepected back. Instead got {len(results)}.")
    earthfile = results[0]
    io.write_debug("Earthfile", earthfile, "dockerfile_to_earthfile")
    return earthfile
