from textwrap import dedent

import guidance

from toearthly.core import io, markdown

gpt4 = guidance.llms.OpenAI("gpt-4")

earthly_basics = io.relative_read("data/earthly_docs/basics.md")
earthly_reference = io.relative_read("data/earthly_docs/summary.md")
earthly_tips = io.relative_read("data/earthly_docs/tips.md")

def prompt(earthfile: str, gha: str, files: str) -> str:
    program = guidance(
        dedent(
            """
        {{#system~}}
        Use the below documentation on Earthfiles to do a code conversion task.
        <<Article>>
        {{earthly_basics}}
        {{earthly_tips}}
        <<Article>>

        The tutorial is over. I will now describe the task.

        Task:
        You are given an Earthfile that has incorrect syntax or doesn't conform to best
        practices.
        The Earthfile is based on a GitHub Actions workflow. this is also given and
        should match it as closely as possible.
        The file structure of the solution is also included because in an Earthfile
        files must be explicitly copied into context.
        The mistakes may be using Dockerfile syntax, or not SAVE ARTIFACT for things it
        COPY or there just may be a better way to structure things.
        Possibly files are copied in that do not exist or a target named `base` is used
        even though that is reservered.

        Do not produce the file yet. Instead, describe how you would approach this
        problem. Then go through the Earthfile section by section and discuss any
        changes that need to be made.

        {{~/system}}
        {{#user~}}
        Files:
        ```
        {{files}}
        ```

        Git Hub Actions:
        ```
        {{gha}}
        ```

        Earthfile:
        ```
        {{earthfile}}
        ```
        {{~/user}}
        {{#assistant~}}
        {{gen "discuss" temperature=0 max_tokens=2000}}
        {{~/assistant}}
        {{#user~}}
        Ok, produce the Earthfile in backticks.
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
        earthly_tips=earthly_tips,
        files=files,
        gha=gha,
        earthfile=earthfile,
    )
    io.write_debug("plan.md", out["discuss"], "earthfile_correction")
    io.write_debug("result.md", out["earthfile"], "earthfile_correction")
    results = markdown.extract_code_blocks(out["Earthfile"])
    if len(results) != 1:
        raise ValueError(f"1 Files exepected back. Instead got {len(results)}.")
    earthfile = results[0]
    io.write_debug("Earthfile", earthfile, "earthfile_correction")
    return earthfile
