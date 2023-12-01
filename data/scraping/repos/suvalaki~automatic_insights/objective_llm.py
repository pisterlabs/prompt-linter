from typing import Any, List, Union

from langchain import PromptTemplate, LLMChain
from langchain.chains.transform import TransformChain
from langchain.chains.sequential import SequentialChain
from langchain.output_parsers import PydanticOutputParser
from langchain.sql_database import SQLDatabase
from langchain.llms.base import BaseLanguageModel

from ai.chains.sql.objective_evaluation.name_evaluation.base import *
from ai.chains.sql.objective_evaluation.name_evaluation.database_filter import (
    TableSelectionChain,
)
from ai.chains.sql.objective_evaluation.name_evaluation.bfs_filter import (
    MultipleTablenameRelevanceEvaluationChain,
)
from ai.chains.sql.objective_evaluation.table_llm_objective import (
    create_objective_evaluate_table_chain,
)
from ai.chains.sql.schema_evaluation.table.table_llm import TableEvaluationChain
from ai.chains.sql.schema_evaluation.table.multiple_table_llm import (
    ExpandTableNameExtractor,
    MultipleTableEvaluatorChain,
)
from ai.chains.sql.schema_evaluation.column.column_single import (
    SingleColumnEvaluateDescriptionChain,
    MultipleColumnDescriptionChain,
)
from ai.chains.sql.schema_evaluation.table.table_llm import (
    TableEvaluateDescriptionChain,
    TableEvaluationChain,
)


def default_name_filter_condition(
    evals: List[TableSelectionDetailThought], threshold: float = 2.5
) -> List[str]:
    return [t.table for t in evals if float(t.score) > threshold]


def create_sql_db_evaluator_chain(
    db: SQLDatabase,
    llm: BaseLanguageModel,
    name_filter_condition: Any = lambda x: default_name_filter_condition(x, 0.75),
):
    input_augmentaion = TransformChain(
        input_variables=[],
        output_variables=["tables"],
        transform=lambda x: {"tables": db.get_usable_table_names()},
    )

    name_evaluator = MultipleTablenameRelevanceEvaluationChain(llm=llm, db=db)

    def name_transform(inputs):
        return {
            "filtered_tables": name_filter_condition(inputs["tablename_evaluations"])
        }

    name_filter = TransformChain(
        input_variables=["tablename_evaluations"],
        output_variables=["filtered_tables"],
        transform=name_transform,
    )

    colum_describer = SingleColumnEvaluateDescriptionChain(llm=llm)
    multiple_column_describer = MultipleColumnDescriptionChain(
        llm=llm,
        db=db,
        chain=colum_describer,
    )
    table_describer = TableEvaluateDescriptionChain(llm=llm)
    table_column_describer = TableEvaluationChain(
        column_description_chain=multiple_column_describer,
        table_description_chain=table_describer,
        return_column_descriptions=True,
        output_key="infered_columns",
    )
    multiple_table_evaluator_chain = MultipleTableEvaluatorChain(
        extract_inputs=ExpandTableNameExtractor(input_key="filtered_tables"),
        chain=SequentialChain(
            input_variables=["objective", "table_name"],
            chains=[
                table_column_describer,
                create_objective_evaluate_table_chain(llm=llm),
            ],
            # return_all=True
        ),
    )

    return SequentialChain(
        input_variables=["objective"],
        chains=[
            input_augmentaion,
            name_evaluator,
            name_filter,
            multiple_table_evaluator_chain,
        ],
        return_all=True,
    )


# class SingleSQLObjectiveChain(LLMChain):

#     db: SQLDatabase
#     name_evaluator: MultipleTablenameRelevanceEvaluationChain
#     name_filter_condition: Any
#     table_evaluator: TableEvaluationChain
#     table_filter_condition: Any
#     action_planner: LLMChain
#     sql_builder: LLMChain
#     sql_evaluator: LLMChain

#     # This should pull out the useful tables and then for those
#     # potentially ueful tables do an assessment of them using
#     # the table_evaluator.

#     # 0. Is this a question that can be answered by SQL ?
#     # 1. Given tablenames in the DB guess which are relevant
#     # 2. Given guess of relevant tables do a deep dive into the tables
#     #    using actual table data. Extract the left over useful tables and
#     #    the useful columns.
#     # 3. Given the useable tables and columns plan how you can answer
#     #    the objective
#     # 3.b If the query cant be answered evaluate that here.
#     # 4. Given the tables, columns and plan create a SQL query
#     # 5. Validate the SQL query.
#     # 6. Run the SQL and get the answer.
#     # 7. Using the question, answer and intermediate information
#     #    answer the question.

#     @property
#     def input_keys(self) -> List[str]:
#         return ["objective"]

#     def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
#         inputs = super().prep_inputs(inputs)
#         tables = self.db.get_usable_table_names()
#         inputs["tables"] = tables
#         return inputs

#     def _call(self, inputs, run_manager, *args, **kwargs):

#         inputs = self.prep_inputs(inputs)

#         name_evals = self.name_evaluator.predict(inputs, run_manager)
#         inputs[""]
