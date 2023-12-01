import sympy as sp
import sympy
import sympy as sp
from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI, OpenAIChat
import langchain
from abc import ABC

from CustomExceptions import PrivateAttributeException
from ExerciseAgent import ExerciseAgent

class ExpressionMapper(ABC):
    pass


class SingleExpressionMapper(ExpressionMapper):
    def __init__(self, expression: str):

        def square(x):
            return x*x

        single_expression_mapping = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "initial": "",
                                     "sqrt": sp.sqrt, "square": square
                                     }
        if expression not in single_expression_mapping.keys():
            raise NotImplementedError(f"The current expression of {expression} is not implemented yet. Sorry!")

        self.expression = single_expression_mapping[expression]


class DoubleExpressionMapper(ExpressionMapper):
    def __init__(self, expression: str):
        def divide(a, b):
            return a / b

        double_expression_mapping = {"add": sp.Add, "sub": sp.Subs,
                                     "mul": sp.Mul, "div": divide
                                     }
        if expression not in double_expression_mapping.keys():
            raise NotImplementedError(f"The current expression of {expression} is not implemented yet. Sorry!")

        self.expression = double_expression_mapping[expression]


class CalculusAgent:
    def __init__(self) -> None:
        self.variables: List[sympy.core.symbol.Symbol] = []

        self.expression = None


    def set_variable(self, *variables: str):
        variables_cat = " ".join(variables)
        sp_vars = sp.symbols(variables_cat)
        self.variables_mapping = {variables[i]: sp_vars[i] for i in range(len(sp_vars))}

    def set_expression(self, expr: ExpressionMapper, connector_expr: ExpressionMapper,
                       dependent_var: str, second_var: str = None):

        if dependent_var not in self.variables_mapping.keys():
            raise ValueError(f"There is no variable set with the name of {dependent_var}. "
                             f"Please set it if you want to use it ")

        if second_var is not None and second_var not in self.variables_mapping.keys():
            raise ValueError(f"There is no variable set with the name of {second_var}. "
                             f"Please set it if you want to use it ")

        if isinstance(expr, SingleExpressionMapper):
            converted_expression = expr.expression(self.variables_mapping[dependent_var])

        elif isinstance(expr, DoubleExpressionMapper):
            converted_expression = expr.expression(self.variables_mapping[dependent_var],
                                                   self.variables_mapping[second_var])

        else:
            raise TypeError("expr must be of type ExpressionMapper")

        if self.expression is None:
            self.expression = converted_expression

        elif isinstance(connector_expr, SingleExpressionMapper):
            self.expression = connector_expr.expression(converted_expression)

        elif isinstance(connector_expr, DoubleExpressionMapper):
            self.expression = connector_expr.expression(converted_expression,
                                                        self.expression)

    def set_outer_func_to_current_expression(self, outer_func: SingleExpressionMapper):
        if not isinstance(outer_func, SingleExpressionMapper):
            raise TypeError("outer_func must be of type SingleExpressionMapper")
        self.expression = outer_func.expression(self.expression)

    def set_binary_outer_func_to_current_expression(self, outer_func: DoubleExpressionMapper, var: float):
        if not isinstance(outer_func, DoubleExpressionMapper):
            raise TypeError("outer_func must be of type DoubleExpressionMapper")
        self.expression = outer_func.expression(var, self.expression)

    def differentiate(self, variable: str):
        if variable not in self.variables_mapping.keys():
            raise ValueError(f"There is no variable set with the name of {variable}. "
                             f"Please set it if you want to use it ")

        diff = sp.diff(self.expression, self.variables_mapping[variable])
        return diff


class CalculusChatAgent(ExerciseAgent):
    def __init__(self, model: langchain.llms):
        super().__init__(model)
        self.template = """Du bist ein Tutor für Mathe Schüler der 12. Klasse am Gymnasium in Deutschland. Sie werden dir eine Frage über Ableitungen odder Integralen stellen.
                        Du bekommst die Funktion und die Lösung und sollst auf Basis dieser eine Antwort auf die Frage des Schülers geben. Bitte bleibe konkret, halluziniere nicht und
                        schreibe "Ich weiß es nicht", wenn du dir nicht sicher bist. Motiviere die Schüler und verwende konkrete Beispiele.
                        Es geht um folgende Funktion: f(x) = {aufgabe}. Sie hat folgende Ableitung f'(x) = {loesung}. {verlauf}.
                        Beantworte diese Frage möglichst kurz und verständlich: {frage}"""


