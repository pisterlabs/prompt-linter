import os
import logging
import re
from langchain.prompts import StringPromptTemplate
from executors.hardhat_executor import HardhatExecutor
from editor.base_editor import BaseEditor
from langchain.llms import OpenAI
from tools.code_editor import CodeEditorTooling
from tools.solidity_compiler import SolidityCompilerTooling
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor, Tool
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from langchain import LLMChain
from langchain.agents import load_tools
import string

logger = logging.getLogger(__name__)

bnf_grammar = """Here is the BNF grammar of solidity:
SourceUnit ::= (PragmaDirective | ImportDirective | ContractDefinition)*
PragmaDirective ::= 'pragma' Identifier ([^;]+) ';'
ImportDirective ::= 'import' StringLiteral ('as' Identifier)? ';'
        | 'import' ('*' | Identifier) ('as' Identifier)? 'from' StringLiteral ';'
        | 'import' '{{' Identifier ('as' Identifier)? ( ',' Identifier ('as' Identifier)? )* '}}' 'from' StringLiteral ';'
ContractDefinition ::= ( 'contract' | 'library' | 'interface' ) Identifier
                     ( 'is' InheritanceSpecifier (',' InheritanceSpecifier )* )?
                     '{{' ContractPart* '}}'
ContractPart ::= StateVariableDeclaration | UsingForDeclaration
             | StructDefinition | ModifierDefinition | FunctionDefinition | EventDefinition | EnumDefinition
InheritanceSpecifier ::= UserDefinedTypeName ( '(' Expression ( ',' Expression )* ')' )?
StateVariableDeclaration ::= TypeName ( 'public' | 'internal' | 'private' | 'constant' )* Identifier ('::=' Expression)? ';'
UsingForDeclaration ::= 'using' Identifier 'for' ('*' | TypeName) ';'
StructDefinition ::= 'struct' Identifier '{{'
                     ( VariableDeclaration ';' (VariableDeclaration ';')* ) '}}'
ModifierDefinition ::= 'modifier' Identifier ParameterList? Block
ModifierInvocation ::= Identifier ( '(' ExpressionList? ')' )?
FunctionDefinition ::= 'function' Identifier? ParameterList
                     ( ModifierInvocation | StateMutability | 'external' | 'public' | 'internal' | 'private' )*
                     ( 'returns' ParameterList )? ( ';' | Block )
EventDefinition ::= 'event' Identifier EventParameterList 'anonymous'? ';'
EnumValue ::= Identifier
EnumDefinition ::= 'enum' Identifier '{{' EnumValue? (',' EnumValue)* '}}'
ParameterList ::= '(' ( Parameter (',' Parameter)* )? ')'
Parameter ::= TypeName StorageLocation? Identifier?
EventParameterList ::= '(' ( EventParameter (',' EventParameter )* )? ')'
EventParameter ::= TypeName 'indexed'? Identifier?
FunctionTypeParameterList ::= '(' ( FunctionTypeParameter (',' FunctionTypeParameter )* )? ')'
FunctionTypeParameter ::= TypeName StorageLocation?
VariableDeclaration ::= TypeName StorageLocation? Identifier
TypeName ::= ElementaryTypeName
         | UserDefinedTypeName
         | Mapping
         | ArrayTypeName
         | FunctionTypeName
         | ( 'address' 'payable' )
UserDefinedTypeName ::= Identifier ( '.' Identifier )*
Mapping ::= 'mapping' '(' ElementaryTypeName '=>' TypeName ')'
ArrayTypeName ::= TypeName '[' Expression? ']'
FunctionTypeName ::= 'function' FunctionTypeParameterList ( 'internal' | 'external' | StateMutability )*
                   ( 'returns' FunctionTypeParameterList )?
StorageLocation ::= 'memory' | 'storage' | 'calldata'
StateMutability ::= 'pure' | 'view' | 'payable'
Block ::= '{{' Statement* '}}'
Statement ::= IfStatement | WhileStatement | ForStatement | Block | InlineAssemblyStatement |
            ( DoWhileStatement | PlaceholderStatement | Continue | Break | Return |
              Throw | EmitStatement | SimpleStatement ) ';'
ExpressionStatement ::= Expression
IfStatement ::= 'if' '(' Expression ')' Statement ( 'else' Statement )?
WhileStatement ::= 'while' '(' Expression ')' Statement
PlaceholderStatement ::= '_'
SimpleStatement ::= VariableDefinition | ExpressionStatement
ForStatement ::= 'for' '(' (SimpleStatement)? ';' (Expression)? ';' (ExpressionStatement)? ')' Statement
InlineAssemblyStatement ::= 'assembly' StringLiteral? AssemblyBlock
DoWhileStatement ::= 'do' Statement 'while' '(' Expression ')'
Continue ::= 'continue'
Break ::= 'break'
Return ::= 'return' Expression?
Throw ::= 'throw'
EmitStatement ::= 'emit' FunctionCall
VariableDefinition ::= (VariableDeclaration | '(' VariableDeclaration? (',' VariableDeclaration? )* ')' ) ( '=' Expression )?
Expression
  ::= Expression ('++' | '--')
  | NewExpression
  | IndexAccess
  | MemberAccess
  | FunctionCall
  | '(' Expression ')'
  | ('!' | '~' | 'delete' | '++' | '--' | '+' | '-') Expression
  | Expression '**' Expression
  | Expression ('*' | '/' | '%') Expression
  | Expression ('+' | '-') Expression
  | Expression ('<<' | '>>') Expression
  | Expression '&' Expression
  | Expression '^' Expression
  | Expression '|' Expression
  | Expression ('<' | '>' | '<=' | '>=') Expression
  | Expression ('==' | '!=') Expression
  | Expression '&&' Expression
  | Expression '||' Expression
  | Expression '?' Expression ':' Expression
  | Expression ('=' | '|=' | '^=' | '&=' | '<<=' | '>>=' | '+=' | '-=' | '*=' | '/=' | '%=') Expression
  | PrimaryExpression
PrimaryExpression ::= BooleanLiteral
                  | NumberLiteral
                  | HexLiteral
                  | StringLiteral
                  | TupleExpression
                  | Identifier
                  | ElementaryTypeNameExpression
ExpressionList ::= Expression ( ',' Expression )*
NameValueList ::= Identifier ':' Expression ( ',' Identifier ':' Expression )*
FunctionCall ::= Expression '(' FunctionCallArguments ')'
FunctionCallArguments ::= '{{' NameValueList? '}}'
                      | ExpressionList?
NewExpression ::= 'new' TypeName
MemberAccess ::= Expression '.' Identifier
IndexAccess ::= Expression '[' Expression? ']'
BooleanLiteral ::= 'true' | 'false'
NumberLiteral ::= ( HexNumber | DecimalNumber ) (' ' NumberUnit)?
NumberUnit ::= 'wei' | 'szabo' | 'finney' | 'ether'
           | 'seconds' | 'minutes' | 'hours' | 'days' | 'weeks' | 'years'
HexLiteral ::= 'hex' ('"' ([0-9a-fA-F]{{2}})* '"' | '\'' ([0-9a-fA-F]{{2}})* '\'')
StringLiteral ::= '"' ([^"\r\n\\] | '\\' .)* '"'
Identifier ::= [a-zA-Z_$] [a-zA-Z_$0-9]*
HexNumber ::= '0x' [0-9a-fA-F]+
DecimalNumber ::= [0-9]+ ( '.' [0-9]* )? ( [eE] [0-9]+ )?
TupleExpression ::= '(' ( Expression? ( ',' Expression? )*  )? ')'
                | '[' ( Expression  ( ',' Expression  )*  )? ']'
ElementaryTypeNameExpression ::= ElementaryTypeName
ElementaryTypeName ::= 'address' | 'bool' | 'string' | Int | Uint | Byte | Fixed | Ufixed
Int ::= 'int' | 'int8' | 'int16' | 'int24' | 'int32' | 'int40' | 'int48' | 'int56' | 'int64' | 'int72' | 'int80' | 'int88' | 'int96' | 'int104' | 'int112' | 'int120' | 'int128' | 'int136' | 'int144' | 'int152' | 'int160' | 'int168' | 'int176' | 'int184' | 'int192' | 'int200' | 'int208' | 'int216' | 'int224' | 'int232' | 'int240' | 'int248' | 'int256'
Uint ::= 'uint' | 'uint8' | 'uint16' | 'uint24' | 'uint32' | 'uint40' | 'uint48' | 'uint56' | 'uint64' | 'uint72' | 'uint80' | 'uint88' | 'uint96' | 'uint104' | 'uint112' | 'uint120' | 'uint128' | 'uint136' | 'uint144' | 'uint152' | 'uint160' | 'uint168' | 'uint176' | 'uint184' | 'uint192' | 'uint200' | 'uint208' | 'uint216' | 'uint224' | 'uint232' | 'uint240' | 'uint248' | 'uint256'
Byte ::= 'byte' | 'bytes' | 'bytes1' | 'bytes2' | 'bytes3' | 'bytes4' | 'bytes5' | 'bytes6' | 'bytes7' | 'bytes8' | 'bytes9' | 'bytes10' | 'bytes11' | 'bytes12' | 'bytes13' | 'bytes14' | 'bytes15' | 'bytes16' | 'bytes17' | 'bytes18' | 'bytes19' | 'bytes20' | 'bytes21' | 'bytes22' | 'bytes23' | 'bytes24' | 'bytes25' | 'bytes26' | 'bytes27' | 'bytes28' | 'bytes29' | 'bytes30' | 'bytes31' | 'bytes32'
Fixed ::= 'fixed' | ( 'fixed' [0-9]+ 'x' [0-9]+ )
Ufixed ::= 'ufixed' | ( 'ufixed' [0-9]+ 'x' [0-9]+ )
AssemblyBlock ::= '{{' AssemblyStatement* '}}'
AssemblyStatement ::= AssemblyBlock
                  | AssemblyFunctionDefinition
                  | AssemblyVariableDeclaration
                  | AssemblyAssignment
                  | AssemblyIf
                  | AssemblyExpression
                  | AssemblySwitch
                  | AssemblyForLoop
                  | AssemblyBreakContinue
AssemblyFunctionDefinition ::=
    'function' Identifier '(' AssemblyIdentifierList? ')'
    ( '->' AssemblyIdentifierList )? AssemblyBlock
AssemblyVariableDeclaration ::= 'let' AssemblyIdentifierList ( ':=' AssemblyExpression )?
AssemblyAssignment ::= AssemblyIdentifierList ':=' AssemblyExpression
AssemblyExpression ::= AssemblyFunctionCall | Identifier | Literal
AssemblyIf ::= 'if' AssemblyExpression AssemblyBlock
AssemblySwitch ::= 'switch' AssemblyExpression ( Case+ AssemblyDefault? | AssemblyDefault )
AssemblyCase ::= 'case' Literal AssemblyBlock
AssemblyDefault ::= 'default' AssemblyBlock
AssemblyForLoop ::= 'for' AssemblyBlock AssemblyExpression AssemblyBlock AssemblyBlock
AssemblyBreakContinue ::= 'break' | 'continue'
AssemblyFunctionCall ::= Identifier '(' ( AssemblyExpression ( ',' AssemblyExpression )* )? ')'
AssemblyIdentifierList ::= Identifier ( ',' Identifier )*
"""

template = """You're an experienced smart contract developer.

You are asked to complement codes of smart contract in Solidity for certain use case.
You are given use case, requirements, specifications, and the current codes.
You have access to a Code Editor and a compiler, that can be used through the following tools:

{tools}

You can import only the contracts by OpenZeppelin, URL of the documentation is https://docs.openzeppelin.com/contracts/4.x/.
You must not change the implementation in case of no build errors.
You must declare import statement outside of contract definition.
You must write codes which can be compiled by version 0.8.0 solidity compiler.
You should ALWAYS be careful not to have syntax errors.
You should ALWAYS fix syntax error when you find.
You should ALWAYS think what to do next.

Use the following format:

Use-case: the input you must implement
Requirements: the requirements of the contract
Current Source Code: Your current code state that you are editing
Thought: you should always think about what to code next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: The result of your last action
... (this Thought/Action/Action Input/Source Code/Code Result can repeat N times)

Thought: I have finished the task
Task Completed: the task has been implemented

Example task:
Task: the input task you must implement

Thought: To start, we need to add the line of code for constructor
Action: CodeEditorAddCode
Action Input: 
1
constructor() {{
}}
Observation:None

Thought: I have added constructor to the codes. I should compile the code to check the codes are correct
Action: CompileSolidity
Action Input: 

Observation: Succeeded

Thought: The codes are correct, it has constructor now.
Action: None
Action Input:
Output is correct

Observation:None is not a valid tool, try another one.

Thought: I have concluded that the output is correct
Task Completed: the task is completed.

Now we begin with a real task!

Use-case: {query}

Requirements: {requirements}

Specifications: {specifications}

Source Code: {source_code}

{agent_scratchpad}

Thought:"""

class CodeEditorPromptTemplate(StringPromptTemplate):
    template: str
    editor: BaseEditor
    tools: List[Tool]
    requirements: str
    specifications: str

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        kwargs["agent_scratchpad"] = thoughts
        kwargs["source_code"] = self.editor.get_code().replace('{', '{{').replace('}', '}}')
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["requirements"] = self.requirements
        kwargs["specifications"] = self.specifications

        return self.template.format(**kwargs)

class CodeEditorOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print("llm output: ", llm_output, "end of llm output")

        if "Task Completed:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2)

        logger.info('action=%s, action_input=%s', action, action_input.strip(" ").strip('"'))

        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

class FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

class AgentRunner:
    def __init__(self, hardhat, contract_path):
        logger.info("AgentRunner::__init__ contract_path=%s", contract_path)

        self.hardhat = hardhat
        self.contract_path = contract_path

    def execute(self, query, requirements, specifications):
        logger.info("AgentRunner::execute")

        editor = BaseEditor(self.contract_path)
        editor.load_code()

        code_edit_tooling = CodeEditorTooling(editor)
        solidity_compiler_tooling = SolidityCompilerTooling(self.hardhat, editor)

        model = OpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0.0)

        tools = [
            code_edit_tooling.build_add_codes_tool(),
            code_edit_tooling.build_change_code_tool(),
            code_edit_tooling.build_delete_codes_tool(),
            solidity_compiler_tooling.build_compile_tool()
        ] + load_tools(["serpapi"], llm=model)

        prompt = CodeEditorPromptTemplate(
            template=template,
            editor=editor,
            tools=tools,
            input_variables=["query", "intermediate_steps"],
            requirements=requirements,
            specifications=specifications.replace('{', '{{').replace('}', '}}')
        )

        tool_names = [tool.name for tool in tools]

        llm_chain = LLMChain(llm=model, prompt=prompt)
        output_parser = CodeEditorOutputParser()

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )

        agent_executor.run(query)

        editor.save_code()
        self.hardhat.format()
