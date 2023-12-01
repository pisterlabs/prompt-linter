from textwrap import dedent
import contextlib
from toearthly.core import io, markdown, constants
import guidance
from typing import Tuple

gpt4 = guidance.llms.OpenAI("gpt-4")

input1 = io.relative_read("data/python_lint/workflow.yml")
cot1 = io.relative_read("data/python_lint/gha_to_bash_prompt_plan.md")
result1 = io.relative_read("data/python_lint/gha_to_bash_prompt_result.md")

input2 = io.relative_read("data/docker_simple/workflow.yml")
cot2 = io.relative_read("data/docker_simple/gha_to_bash_prompt_plan.md")
result2 = io.relative_read("data/docker_simple/gha_to_bash_prompt_result.md")

def call_identify(identify, *args, **kwargs):
    with open(constants.DEBUG_DIR + "log.txt", 'a') as f, \
            contextlib.redirect_stdout(f), \
            contextlib.redirect_stderr(f):
        return identify(*args, **kwargs)

# Seems like we should pass in file structure as well?
def prompt1(gha : str, files: str) -> Tuple[str, str, str]:
 
    identify = guidance(dedent('''
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
    '''), llm=gpt4)
    
    out = call_identify(identify,
        gha=dedent(gha), 
        files=files, 
        input1=input1, 
        cot1=cot1, 
        result1=result1, 
        input2=input2, 
        cot2=cot2, 
        result2=result2)
    io.write_debug("gha_to_bash_prompt_plan.md", out["discuss"])
    io.write_debug("gha_to_bash_prompt_result.md", out["files"])
    results = markdown.extract_code_blocks(out["files"])
    if len(results) != 3:
        raise ValueError(f"3 Files exepected back. Instead got {len(results)}")
    io.write_debug("run.sh", results[0])
    io.write_debug("build.Dockerfile", results[1])
    io.write_debug("build.sh", results[2])
    return (results[0],results[1], results[2])

earthly_basics = io.relative_read("data/earthly_docs/basics.md") 
earthly_reference = io.relative_read("data/earthly_docs/summary.md") 
earthly_tips = io.relative_read("data/earthly_docs/tips.md") 

input1 = io.relative_read("data/python_lint/files.md")
cot1 = io.relative_read("data/python_lint/EarthfilePlan.md")
result1 = io.relative_read("data/python_lint/Earthfile")

def prompt2(files: str, run : str, docker : str, build : str) -> str:
    identify = guidance(dedent('''
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
    '''), llm=gpt4)
    out = call_identify(identify, earthly_basics=earthly_basics, 
                       input1=input1, 
                       cot1=cot1, 
                       result1=result1, 
                       files=files, 
                       run=run,
                       docker=docker, 
                       build=build)
    io.write_debug("EarthfilePlan.md", out["discuss"])
    results = markdown.extract_code_blocks(out["Earthfile"])
    if len(results) != 1:
        raise ValueError(f"1 Files exepected back. Instead got {len(results)}.")
    earthfile = results[0]
    io.write_debug("Earthfile.1",earthfile)
    return earthfile

def prompt3(earthfile: str, gha : str, files: str) ->  str:
    identify = guidance(dedent('''
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

    '''), llm=gpt4)
    out = call_identify(identify,
                   earthly_basics=earthly_basics, 
                   earthly_tips=earthly_tips, 
                   input1=input1, 
                   files=files, 
                   gha=gha,
                   earthfile=earthfile)
    io.write_debug("EarthfileFixPlan.md", out["discuss"])
    results = markdown.extract_code_blocks(out["Earthfile"])
    if len(results) != 1:
        raise ValueError(f"1 Files exepected back. Instead got {len(results)}.")
    return results[0]