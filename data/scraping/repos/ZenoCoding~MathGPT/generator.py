import jsonlines
import utils
from utils import ProblemType
import random
from enum import Enum

from solver import Solver

import openai

import sympy as sp

openai.api_key = "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"

# Function that generates a random coefficient that isn't 1
def random_coefficient() -> int:
    return random.choice([i for i in range(-10, 10) if i not in [1, 0]])

# Function that generates simple binomials with variable x
def generate_binomial(leading_is_one: bool = True) -> str:
    return f"({random_coefficient() if not leading_is_one else ''}x {random.Random().choice(['+', '-'])} {random.Random().randint(1, 10)})"


# Function that generates simple factor able polynomials with variable x by multiplying two binomials
def generate_factorable_polynomial(degree: int = 1, leading_is_one: bool = True) -> str:
    polynomial = utils.convert_expression(generate_binomial(leading_is_one))
    x = random.Random().randint(0, 5)
    for i in range(degree-1):
        # generate a binomial
        binomial = utils.convert_expression(generate_binomial(leading_is_one))
        polynomial = polynomial * binomial
        # On the last iteration, occasionally add a partially factored portion to the polynomial
        if i == degree - 2 or i == degree - 3:
            if x == 0:
                pass
            elif x == 1 and not leading_is_one:
                polynomial *= random_coefficient()
            else:
                polynomial = sp.expand(polynomial)
        else:
            polynomial = sp.expand(polynomial)

    return str(polynomial)

def generate_random_polynomial(degree: int = 1, leading_is_one: bool = True) -> str:

    r = random.Random()
    result = []
    result.append(random_coefficient()) if not leading_is_one else result.append(1)
    for _ in range(degree):
        result.append(r.randint(-10, 10))
    x = sp.symbols('x')
    result = sp.sympify((sp.Poly.from_list(result, gens=x)).as_expr())

    return str(result)


def generate_equation(degree: int = 2, leading_is_one: bool = True) -> sp.Equality:
    r = random.Random()
    poly = generate_random_polynomial(degree, leading_is_one)
    n = r.randint(0, 2)
    if n == 0:
        # polynomial on the other side (degree less than or equal to current degree)
        result = f"{poly} = {generate_random_polynomial(r.randint(1, degree), leading_is_one)}"
    else:
        # 0 on the other side
        result = f"{poly} = 0"

    return result


prompts = {
    "factor_polynomial": [
        f"Factor the polynomial %s.",
        f"Find the factors of the polynomial %s.",
        f"Find the factorization of the polynomial %s.",
        f"%s Factor.",
        f"Factor %s",
        f"Factor the polynomial %s",
        f"Please factor the polynomial %s"
    ],

    "solve": [
        f"Solve for x. %s.",
        f"Find the roots of %s.",
        "Solve. %s",
        "Algebraically solve %s",
        "%s Solve the equation.",
    ],
    
    "solve_quadratic": [
        f"Solve for x. %s.",
        f"Find the roots of %s.",
        f"Use the quadratic equation to solve for x %s.",
        f"Solve. %s.",
        f"%s Solve.",
        f"equate to 0 and solve %s",
        f"Please factor the polynomial %s",
    ],

    "non-generate-able": [
    ]

}


class ProblemType(Enum):

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, prompts: list, expression_generator, solver):
        # Name of the problem type (id)

        # List of prompts to choose from, should contain placeholders for the expression - %s
        self.prompts = prompts

        # Function that generates an sp.Expr expression
        self.expression_generator = expression_generator

        # Function that solves an expression
        self.solver = solver

    def generate_problem(self, amount: int = 1) -> list[tuple[str]]:
        result = []
        for _ in range(amount):
            # Choose a random prompt
            prompt = random.Random().choice(self.prompts)
            # Generate an expression
            expression = self.expression_generator()
            # Replace the placeholder with the expression
            prompt = prompt.replace("%s", expression)
            # Return the prompt and the expression
            result.append((prompt, expression))
            print(f"Generating prompt | Type: {self.name.lower()} |  {_ + 1}/{AMOUNT} {round((_ + 1) / AMOUNT * 100, 2)}%")
        return result

    def solve(self, expression: str):
        return self.solver(expression)


    # Degree of two or less
    FACTOR_SIMPLE_POLYNOMIAL = prompts["factor_polynomial"], lambda: generate_factorable_polynomial(2, True), Solver.factor  # x^2 + 3x + 1
    FACTOR_SIMPLE_NONONE_POLYNOMIAL = prompts["factor_polynomial"], lambda: generate_factorable_polynomial(2, False), Solver.factor  # 3x^2 + 3x + 1

    # Degree more than two
    FACTOR_HIGH_DEGREE_POLYNOMIAL = prompts["factor_polynomial"], lambda: generate_factorable_polynomial(random.Random().randint(3, 10), True), Solver.factor  # x^3 + 3x + 1

    # Lower than 1 degree polynomial
    FACTOR_SIMPLE = prompts["factor_polynomial"], lambda: generate_factorable_polynomial(1, False), Solver.factor  # 2x+2

    SOLVE_SIMPLE = prompts["solve"], lambda: generate_equation(1, False), Solver.solve  # 2x+2 = 0

    QUADRATIC_POLYNOMIAL_ROOT_QUADRATIC_EQUATION = prompts["solve_quadratic"], lambda: generate_equation(2, False), Solver.solve
    # x^2 + 2x + 1 = 0

    # Arithmetic
    INTEGER_ADDITION = prompts["non-generate-able"], lambda: None, Solver.simplify

    INTEGER_ADDITION_WORD = prompts["non-generate-able"], lambda: None, Solver.simplify

    EVALUATE_INTEGER_ADDITION = prompts["non-generate-able"], lambda: None, Solver.evaluate

    INTEGER_SUBTRACTION = prompts["non-generate-able"], lambda: None, Solver.simplify

    INTEGER_SUBTRACTION_WORD = prompts["non-generate-able"], lambda: None, Solver.simplify

    EVALUATE_INTEGER_SUBTRACTION = prompts["non-generate-able"], lambda: None, Solver.evaluate

    INTEGER_MULTIPLICATION = prompts["non-generate-able"], lambda: None, Solver.simplify

    INTEGER_MULTIPLICATION_WORD = prompts["non-generate-able"], lambda: None, Solver.simplify

    EVALUATE_INTEGER_MULTIPLICATION = prompts["non-generate-able"], lambda: None, Solver.evaluate

    DIVISION = prompts["non-generate-able"], lambda: None, Solver.simplify

    DIVISION_WORD = prompts["non-generate-able"], lambda: None, Solver.simplify

    EVALUATE_DIVISION = prompts["non-generate-able"], lambda: None, Solver.evaluate

    PLACE_VALUE_NUMBER_OF_ONES = prompts["non-generate-able"], lambda: None, lambda params: Solver.place_value_number_of_digit(params, 1)

    PLACE_VALUE_NUMBER_OF_TENS = prompts["non-generate-able"], lambda: None, lambda params: Solver.place_value_number_of_digit(params, 2)

    ADDITION_WITH_PLACE_VALUE = prompts["non-generate-able"], lambda: None, Solver.simplify

    SUBTRACTION_WITH_PLACE_VALUE = prompts["non-generate-able"], lambda: None, Solver.simplify

AMOUNT = 30


def generate_prompts():
    for e in ProblemType:
        if len(e.prompts) == 0:
            print(f"Skipping generation for {e.name}...")
            continue

        print(f"Generating prompts for {e.name}...")
        for i in e.generate_problem(AMOUNT):
            prompt, polynomial = i
            with jsonlines.open("verifiers/poly_v1.jsonl", mode="a") as writer:
                writer.write({"prompt": prompt + "\n\n###\n\n",
                              "completion": f" {polynomial} | {e.name.lower()}###"})


# function that takes the prompts and "randomizes" them, converting some "**" to ^ and removing some cases of
# explicit multiplication and spaces
def randomize_prompts():
    with jsonlines.open("output/prompts.jsonl", mode="r") as reader:
        new_prompts = []
        for prompt in reader:
            result = ""
            r = random.Random()
            for c in range(len(prompt["prompt"])):
                if r.randint(0, 1) == 1:
                    if prompt["prompt"][c] == '*' and prompt["prompt"][c + 1] == "*":
                        result += "^"
                    elif prompt["prompt"][c] == "*" and prompt["prompt"][c - 1] != "*":
                        result += ""
                    else:
                        result += prompt["prompt"][c]
                else:
                    result += prompt["prompt"][c]
            new_prompts.append({"prompt": result, "completion": prompt["completion"]})

        for prompt in new_prompts:
            with jsonlines.open("output/prompts.jsonl", mode="a") as writer:
                writer.write({"prompt": prompt["prompt"], "completion": prompt["completion"]})

#randomize_prompts()

#export OPENAI_API_KEY="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"

# openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
# openai api fine_tunes.create -t formatted_prompts.jsonl -m babbage --suffix "recognition"
