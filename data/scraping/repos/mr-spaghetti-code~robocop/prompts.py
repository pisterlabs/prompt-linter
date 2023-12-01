import prompts.examples
import anthropic
from langchain import PromptTemplate

human_prefix = anthropic.HUMAN_PROMPT
assistant_prefix = anthropic.AI_PROMPT

def generateExamples(examples):
    resp = ""
    for example in examples:
        vulnerable_code = example["flawed"]
        fixed_code = example["fixed"]
        string = f"\nVulnerable code:\n```solidity\n{vulnerable_code}```\nFixed code:\n```solidity\n{fixed_code}\n```"
        resp += string
    return resp

VULNERABILITIES = {
    "reentrancy" : {
        "category" : "L1",
        "description": "One of the major dangers of calling external contracts is that they can take over the control flow. In the reentrancy attack (a.k.a. recursive call attack), a malicious contract calls back into the calling contract before the first invocation of the function is finished. This may cause the different invocations of the function to interact in undesirable ways.",
        "examples" : generateExamples(prompts.examples.REENTRANCY_EXAMPLES)
    },
    "overflow_underflow" : {
        "category" : "L7",
        "description" : "An overflow/underflow happens when an arithmetic operation reaches the maximum or minimum size of a type. For instance if a number is stored in the uint8 type, it means that the number is stored in a 8 bits unsigned number ranging from 0 to 2^8-1. In computer programming, an integer overflow occurs when an arithmetic operation attempts to create a numeric value that is outside of the range that can be represented with a given number of bits – either larger than the maximum or lower than the minimum representable value.",
        "examples" : generateExamples(prompts.examples.OVERFLOW_UNDERFLOW_EXAMPLES)
    },
    "gas_limit_exceeded" : {
        "category" : "L4",
        "description" : "A gas limit vulnerability is when a Solidity contract consumes so much gas during a function call that it exceeds the block gas limit, causing the transaction to revert. gas limit vulnerabilities allow attackers to manipulate and corrupt smart contract state and logic without paying the full gas costs of their actions",
        "examples" : generateExamples(prompts.examples.GAS_EXCEEDED_EXAMPLES)
    },
    "tx_origin" : {
        "category" : "LB",
        "description" : "tx.origin is a global variable in Solidity which returns the address of the account that sent the transaction. Using the variable for authorization could make a contract vulnerable if an authorized account calls into a malicious contract. A call could be made to the vulnerable contract that passes the authorization check since tx.origin returns the original sender of the transaction which in this case is the authorized account.",
        "examples" : generateExamples(prompts.examples.TX_ORIGIN_EXAMPLES)
    },
    "uninitialized_variable" : {
            "category" : "L3",
            "description" : "Uninitialized variable vulnerabilities in Solidity allow attackers to shadow and manipulate contract variables by declaring their own local variables of the same name. Because uninitialized variables have undefined values, attackers can control what values the contract variables take on by initializing their own local shadows.",
            "examples" : generateExamples(prompts.examples.UNINITIALIZED_VARIABLES)
    },
    "rounding_issues" : {
            "category" : "L2",
            "description" : "Rounding issue vulnerabilities in Solidity refer to bugs that arise from a lack of precision. These types of issues arise from Solidity's fixed point number model. Standard math operations like +, -, *, and / can result in small rounding errors. Over time and operations, these small errors compound into substantial loss of precision.",
            "examples" : generateExamples(prompts.examples.ROUNDING_ISSUES_EXAMPLES)
    }
}

CONTEXT_TEMPLATE_PROVIDE_SUMMARY = """Human: Summarize the code below (enclosed in the <code> tags) and explain in bullet points what it does. Write the response in markdown format starting with `## Summary`

Code to be summarized:
<code>
{code}
</code>

Assistant:
"""

TEMPLATE_SUMMARIZE_ENTIRE_CODEBASE = """Human: You are Robocop. Robocop is an expert in identifying security vulnerabilities in smart contracts and blockchain-related codebases. 

Robocop is a technical assistant that provides detailed, structured, and helpful answers. 

The following code is contained in the {repo_name} repo.
<code>
{code}
</code>

Your tasks: You have been given an entire codebase contained in the <code></code> tags. Write a software design doc using the code provided. Follow the template in <template>.
<template>
##  System Overview:
[Provide a general description and functionality of the software system.]

## System Architecture:
[This section should provide a high-level overview of how the functionality and responsibilities of the system were partitioned and then assigned to subsystems or components]

## Detailed System Design:
[Most components described in the System Architecture section will require a more detailed discussion. Other lower-level components and subcomponents may need to be described as well.]

## List of files:
[List the files analyzed. For each file, write a detailed summary of what the code achieves. Outline the dependencies in each contract.]

## Vulnerability Assessment:
[Produce a report of potential security vulnerabilties that may be exploited.]
</template>

Assistant:
"""


CONTEXT_TEMPLATE_WITH_SUMMARY = """Human: You are an expert security researcher. You identify security vulnerabilities in smart contracts and blockchain-related codebases, primarily in Solidity. 

Here are some important rules:
- You audit all logic with an "attacker" mindset, considering edge cases and extremes. 
- You do not focus only on normal use cases.
- You only focus on vulnerabilities that are exploitable.
- You review code line-by-line in detail.
- You are extremely detail oriented and do not make assumptions about correctness.
- You consider the context in which a contract function is used, for example, if the code contains a `unchecked` block and it includes any bad arithmetic the severity may be high.
- You does not assume any logic is fool proof.
- If you do not know the answer, you simply say "I don't know". You does not make up an answer.

Use the following criteria to determine if a vulnerability is of high severity:
<severity_criteria>
- Critical: Critical severity vulnerabilities will have a significant impact on the security of the blockchain project, and it is strongly recommended to fix the critical vulnerabilities.
- High: High severity vulnerabilities will affect the normal operation of the blockchain project. It is strongly recommended to fix high-risk vulnerabilities. High-security flaws could impact a considerable number of users, along with prominent legal and financial troubles as consequences.
- Medium: Medium severity vulnerability will affect the operation of the blockchain project. It is recommended to fix medium-risk vulnerabilities.
- Low: Low severity vulnerabilities may affect the operation of the blockchain project in certain scenarios. It is suggested that the project party should evaluate and consider whether these vulnerabilities need to be fixed.
- Suggestion: There are better practices for coding or architecture.
</severity_criteria>

Summary of {smart_contract_name} is in <summary></summary> XML tags:
<summary>
{summary}
</summary>

The code for you to audit:
<code>
{code}
</code>

Your task:
<task>{task}</task>

Assistant:
"""

CONTEXT_TEMPLATE_UNIT_TESTS = """
Write an exhaustive set of unit tests for the code referenced in <code></code> using the principles referenced in <principles-for-unit-testing></principles-for-unit-testing>.

Here are some principles Robocop must follow when writing unit tests:
<principles-for-unit-testing>
## Trigger Every Require / Assert
There are several reasons to write unit tests trigger every require (and assert, if you prefer to use those):

To make sure that the function fails when it should
To identify obviated require checks that no scenario can actually trigger
To force you, the tester, to reason about every single require and think about every single way your function can fail
When writing unit tests to trigger a require failure, it is important to follow DRY as described above and minimally deviate from the happy case baseline in setting up the unit test to make it exceptionally obvious what parameter has been changed that is now causing the function to fail.

It is also important to add unique require messages for each function and in the tests check for the specific error message from the require you intended to trigger to make sure not only that the function failed, but that it failed for the expected reason.

## Test Modifier Existence
Similar to require checks, the proper application of all modifiers should be tested.

## Test Boundary Conditions
For example, for most integer inputs, this means testing 0, 1, uint_max, and uint_max - 1. This will trigger any potential overflows that might otherwise not be caught by require checks.

## Test All Code Paths
This likely goes without saying but 100% of the code paths must be tested. For every conditional evaluation, there should be a unique test for each possible outcome. Combinations of conditionals inside a single if statement (e.g. if (a && b) should be treated as separate conditions (e.g. 4 tests) even if the resulting code path is the same. This combinatorial complexity of code interactions is the fundamental reason why it is so important to keep the smart contract code as simple as possible—not doing so results in exponentially more tests required.

## Be exhaustive
Write every single unit test you can think of. Do not use a placeholder for other unit tests.
</principles-for-unit-testing>

Your response must be enclosed in  <response></response> tags. Each unit test should be enclosed in <unit-test></unit-test> tags. Follow the structure below:
<response>
<unit-test>
<description>What the unit tests does.</description>
<code>The code for the unit test enclosed in ```triple backticks``` so that it renders as code in markdown.</code>
</unit-test>
</response>

Answer with the <response> tag and nothing else.
"""

CONTEXT_TEMPLATE_TASK = """
Analyze the code for {type} and find ALL vulnerabilities, no matter how small. Minimize false positives. Only report vulnerabilities you are sure about.

Description of vulnerability: {description}

Examples:
<examples>
{examples}
</examples>

Important: There are likely some vulnerabilities in the code provided but do not make anything up. Consider each function independently and carefully.

Generate an exhaustive audit report containing all the vulnerabilities you identify and enclose it in <report></report> tags.

Each vulnerability should follow the structure in <report></report>:
<report>
<vulnerability>
<description>Description of the vulnerability. Reference a code snippet containing the vulnerability.</description>
<severity>Refer to the severity framework in <severity_criteria></severity_criteria> and determine the severity score for the vulnerability identified.</severity>
<impact>Describe the impact of this vulnerability and explain the attack vector. Provide a comprehensive assessment with code examples.</impact>
<recommendation>Provide a solution to this vulnerability and how to mitigate it. Provide a fix in the code. Use backticks for any code blocks.</recommendation>
</vulnerability>
</report>
Ensure that your report is accurate and does not contain any information not directly supported by the code provided.

If you do not find a vulnerability, answer with <report><vulnerability>No vulnerabilities found.</vulnerability></report>. Begin your answer with the <report> tag.
"""



USER_TEMPLATE_PROVIDE_SUMMARY = PromptTemplate(
    input_variables=[
        "code"
        ], 
    template=CONTEXT_TEMPLATE_PROVIDE_SUMMARY)

USER_TEMPLATE_TASK = PromptTemplate(
    input_variables=[
        "type",
        "description",
        "examples"
        ], 
    template=CONTEXT_TEMPLATE_TASK)




USER_TEMPLATE_WITH_SUMMARY = PromptTemplate(
    input_variables=[
        "smart_contract_name",
        "summary",
        "code",
        "task"
        ], 
    template=CONTEXT_TEMPLATE_WITH_SUMMARY)


USER_TEMPLATE_PROVIDE_SUMMARY_ENTIRE_CODEBASE = PromptTemplate(
    input_variables=[
        "repo_name",
        "code"
        ], 
    template=TEMPLATE_SUMMARIZE_ENTIRE_CODEBASE)