from langchain.agents import Tool

class SolidityCompilerTooling():
    def __init__(self, hardhat_executor, code_editor):
        self.hardhat_executor = hardhat_executor
        self.code_editor = code_editor

    def build_compile_tool(self):
        return Tool(
            name="CompileSolidity",
            func=self._compile,
            description="""Use to compile codes and get compile errors which are useful for code fix.
Example 1 (success case):

Action: CompileSolidity
Action Input: None
Observation: Succeeded
Thought: In this example, there is no compile error.

Example 2 (failure example):

Action: CodeEditorAddCode
Action Input: None
Observation: Failure
DeclarationError: Undeclared identifier. Did you mean "stealthMetaPublicKey"?
 --> worker/Test.sol:7:9:
  |
7 |         stealthMetaPublicK[msg.sender] = publicKey;
  |         ^^^^^^^^^^^^^^^^^^


Error HH600: Compilation failed

Thought: In this example, there are wrong syntax at line 7. 
""",
        )

    def _compile(self, query: str) -> str:
        self.code_editor.save_code()
        format_result = self.hardhat_executor.format()
        self.code_editor.load_code()

        # if format_result is not None:
        #     return 'Failure\n' + format_result

        compile_result = self.hardhat_executor.compile()
        if compile_result is not None:
            return 'Failure\n' + compile_result

        return 'Succeeded'
