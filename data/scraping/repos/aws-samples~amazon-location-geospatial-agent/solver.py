import json

import networkx
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from pydispatch import dispatcher

from geospatial_agent.agent.geospatial.solver.constants import NODE_TYPE_ATTRIBUTE, NODE_TYPE_OPERATION
from geospatial_agent.agent.geospatial.solver.op_graph import OperationsParser, OperationNode
from geospatial_agent.agent.geospatial.solver.prompts import operation_code_gen_intro, \
    operation_task_prefix, operation_reply_example, operation_code_gen_prompt_template, \
    operation_pydeck_example, operation_requirement_gen_task_prefix, predefined_operation_requirements, \
    shim_instructions
from geospatial_agent.agent.shared import SIGNAL_OPERATION_CODE_GENERATED, SENDER_GEOSPATIAL_AGENT, AgentSignal, \
    EventType, SIGNAL_TAIL_CODE_GENERATED
from geospatial_agent.shared.prompts import HUMAN_ROLE, ASSISTANT_ROLE, HUMAN_STOP_SEQUENCE
from geospatial_agent.shared.shim import get_shim_imports
from geospatial_agent.shared.utils import extract_code, extract_content_xml

from typing import List


class OperationCodeGenOutput:
    def __init__(self,
                 operation_prompt: str,
                 operation_code_gen_response: str,
                 operation_code: str):
        self.operation_prompt = operation_prompt
        self.operation_code_gen_response = operation_code_gen_response
        self.operation_code = operation_code


class InvalidStateError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class Solver:
    def __init__(self,
                 llm: LLM,
                 graph: networkx.DiGraph,
                 graph_code: str,
                 session_id: str,
                 storage_mode: str,
                 task_definition: str,
                 task_name: str,
                 data_locations_instructions: str):
        self.llm = llm
        self.graph = graph
        self.graph_code = graph_code
        self.session_id = session_id
        self.storage_mode = storage_mode
        self.task_def = task_definition
        self.task_name = task_name
        self.data_locations_instructions = data_locations_instructions
        self.operation_parser = OperationsParser(graph)

    def solve(self):
        op_nodes = self.operation_parser.operation_nodes

        for idx, op_node in enumerate(op_nodes):
            operation_code_gen_output = self.gen_operation_code(op_node)
            dispatcher.send(signal=SIGNAL_OPERATION_CODE_GENERATED,
                            sender=SENDER_GEOSPATIAL_AGENT,
                            event_data=AgentSignal(
                                event_source=SENDER_GEOSPATIAL_AGENT,
                                event_message=f"{idx + 1} / {len(op_nodes)}: Generated code for operation {op_node.node_name}",
                                event_data=operation_code_gen_output.operation_code,
                                event_type=EventType.PythonCode
                            ))

            # INFO: Updating Operation Nodes with generated code
            op_node.operation_prompt = operation_code_gen_output.operation_prompt
            op_node.code_gen_response = operation_code_gen_output.operation_code_gen_response
            op_node.operation_code = operation_code_gen_output.operation_code

        return op_nodes

    def assemble(self):
        output_node_names = self.operation_parser.output_node_names
        operation_nodes = self.operation_parser.operation_nodes

        # The head end of the code
        head = ""

        # The tail end of the code
        tail = ""

        reverse_graph = self.graph.reverse(copy=True)

        for idx, output_node in enumerate(output_node_names):
            bfs_edges = networkx.bfs_edges(reverse_graph, source=output_node)
            for bfs_edge in bfs_edges:
                from_node_name, _ = bfs_edge
                current_nx_node = self.graph.nodes[from_node_name]

                if current_nx_node.get(NODE_TYPE_ATTRIBUTE, None) == NODE_TYPE_OPERATION:
                    op_node: OperationNode = next(
                        (op_node for op_node in operation_nodes if op_node.node_name == from_node_name), None)

                    head = "\n" + op_node.operation_code + "\n" + head
                    tail = f'{", ".join(op_node.return_names)}={op_node.function_definition}\n' + tail

        # Adding the session id and task name to the code
        tail = f'\nsession_id = "{self.session_id}"\n' + \
               f'task_name = "{self.task_name}"\n' + \
               f'storage_mode = "{self.storage_mode}"\n' + \
               tail

        dispatcher.send(signal=SIGNAL_TAIL_CODE_GENERATED,
                        sender=SENDER_GEOSPATIAL_AGENT,
                        event_data=AgentSignal(
                            event_source=SENDER_GEOSPATIAL_AGENT,
                            event_message=f"Generated final code block.",
                            event_data=tail,
                            event_type=EventType.PythonCode
                        ))

        assembled_code = head + "\n" + tail
        assembled_code = f'{get_shim_imports()}\n{assembled_code}'
        return assembled_code

    def get_operation_requirement(self, op_node: OperationNode) -> list[str]:
        node_name = op_node.node_name

        task_def = self.task_def.strip("\n").strip()
        op_properties = [
            f'The function description is: {op_node.description}',
            f'The type of work done in this function is: {op_node.operation_type}',
            f'This function is one step to solve the question/task: {task_def}'
        ]

        op_properties_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(op_properties)])

        operation_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(predefined_operation_requirements)])

        op_req_gen_prompt_template: PromptTemplate = PromptTemplate.from_template(operation_requirement_gen_task_prefix)

        chain = LLMChain(llm=self.llm, prompt=op_req_gen_prompt_template)
        req_gen_response = chain.run(
            human_role=HUMAN_ROLE,
            operation_req_gen_intro=operation_code_gen_intro,
            operation_name=node_name,
            pre_requirements=operation_requirement_str,
            operation_properties=op_properties_str,
            assistant_role=ASSISTANT_ROLE,
            stop=[HUMAN_STOP_SEQUENCE]
        ).strip()

        operation_requirement_json = extract_content_xml("json", req_gen_response)
        operation_requirement_list: List[str] = json.loads(operation_requirement_json)
        operation_requirement_list = shim_instructions + operation_requirement_list
        return operation_requirement_list

    def gen_operation_code(self, op_node: OperationNode) -> OperationCodeGenOutput:
        operation_requirement_list = self.get_operation_requirement(op_node)

        node_name = op_node.node_name

        # Get ancestors operations functions. For operations that has ancestors, this will also come with LLM
        # generated code for the operations.
        ancestor_op_nodes = self.operation_parser.get_ancestors(node_name)
        ancestor_op_nodes_code = '\n'.join([op_node.operation_code for op_node in ancestor_op_nodes])

        descendant_op_node = self.operation_parser.get_descendants(node_name)
        descendant_op_node_defs = self.operation_parser.stringify_nodes(descendant_op_node)

        pre_requirements = [
            f'The function description is: {op_node.description}',
            f'The function definition is: {op_node.function_definition}',
            f'The function return line is: {op_node.return_line}'
        ]

        operation_requirements_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(pre_requirements + operation_requirement_list)])

        op_code_gen_prompt_template: PromptTemplate = PromptTemplate.from_template(operation_code_gen_prompt_template)
        op_code_gen_prompt = op_code_gen_prompt_template.format(
            human_role=HUMAN_ROLE,
            operation_code_gen_intro=operation_code_gen_intro,
            operation_task_prefix=operation_task_prefix,
            operation_description=op_node.description,
            task_definition=self.task_def.strip("\n").strip(),
            graph_code=self.graph_code,
            data_locations_instructions=self.data_locations_instructions,
            session_id=self.session_id,
            task_name=self.task_name,
            storage_mode=self.storage_mode,
            operation_reply_example=operation_reply_example,
            operation_pydeck_example=operation_pydeck_example,
            operation_requirements=operation_requirements_str,
            ancestor_operation_code=ancestor_op_nodes_code,
            descendant_operations_definition=str(descendant_op_node_defs),
            assistant_role=ASSISTANT_ROLE
        )

        chain = LLMChain(llm=self.llm, prompt=op_code_gen_prompt_template)
        code_gen_response = chain.run(
            human_role=HUMAN_ROLE,
            operation_code_gen_intro=operation_code_gen_intro,
            operation_task_prefix=operation_task_prefix,
            operation_description=op_node.description,
            task_definition=self.task_def.strip("\n").strip(),
            graph_code=self.graph_code,
            data_locations_instructions=self.data_locations_instructions,
            session_id=self.session_id,
            task_name=self.task_name,
            storage_mode=self.storage_mode,
            operation_reply_example=operation_reply_example,
            operation_pydeck_example=operation_pydeck_example,
            operation_requirements=operation_requirements_str,
            ancestor_operation_code=ancestor_op_nodes_code,
            descendant_operations_definition=str(descendant_op_node_defs),
            assistant_role=ASSISTANT_ROLE,
            stop=[HUMAN_STOP_SEQUENCE]
        ).strip()

        operation_code = extract_code(code_gen_response)

        return OperationCodeGenOutput(
            operation_prompt=op_code_gen_prompt,
            operation_code_gen_response=code_gen_response,
            operation_code=operation_code
        )
