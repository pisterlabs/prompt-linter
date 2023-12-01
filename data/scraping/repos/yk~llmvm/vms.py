from typing import Any
import random
import re
from loguru import logger
import openai
import openai.error
import os

import pydantic

from interface import (
    Alloc,
    AllocArray,
    Expr,
    to_vtype,
    Load,
    Icmp,
    Srem,
    Copy,
    Add,
    Mul,
    Call,
    Assign,
    Store,
    Branch,
    BranchCond,
    Return,
    Program,
    GetElementPtr,
    Switch,
)


class VM(pydantic.BaseModel):
    name: str = "VM"
    constants: dict[str, Any]
    registers: dict[str, Any] = {}
    memory: dict[str, Any] = {}
    _program: Program | None = pydantic.PrivateAttr(None)

    class Config:
        ignored_types = (Expr,)

    def _get_from_registers(self, value):
        return self.registers[value]

    def _to_vtype(self, value, vtype):
        return to_vtype(value, vtype)

    def get_value(self, value, vtype=None):
        if isinstance(value, str):
            if value.startswith("%"):
                value = self._get_from_registers(value)
                if isinstance(value, str):
                    logger.debug(f"found string {value}")
                    if m := re.match(r"arr ptr=(\S+) idx=(\S+)", value):
                        ptr, idx = m.groups()
                        logger.debug(f"performing array lookup on {ptr} {idx}")
                        arr = self._get_from_registers(ptr)
                        logger.debug(f"arr = {arr}")
                        value = self._load_array(arr, idx)
                        logger.debug(f"array lookup result = {value}")
            elif value.startswith("@"):
                value = self.constants[value]
        if vtype is not None:
            value = self._to_vtype(value, vtype)
        return value

    def _cmp_with_op(self, op, lhs, rhs) -> bool:
        match op:
            case "eq":
                return lhs == rhs
            case "ne":
                return lhs != rhs
            case "sgt":
                return lhs > rhs
            case "sge":
                return lhs >= rhs
            case "slt":
                return lhs < rhs
            case "sle":
                return lhs <= rhs
        raise NotImplementedError(op)

    def _rem(self, lhs, rhs):
        return lhs % rhs

    def _add(self, lhs, rhs):
        return lhs + rhs

    def _mul(self, lhs, rhs):
        return lhs * rhs

    def _load_array(self, arr, idx):
        return arr[idx]

    def _alloc_array(self, vtype, size):
        if vtype in ("i32", "i8"):
            return {str(i): 0 for i in range(size)}
        raise NotImplementedError(vtype)

    def eval_expr(self, expr: Expr):
        logger.debug(f"eval_expr({expr})")
        match expr:
            case Copy(ptr=ptr):
                return self.get_value(ptr)
            case Alloc(vtype=vtype):
                return self.get_value(0, vtype)
            case AllocArray(vtype=vtype, size=size):
                return self._alloc_array(vtype, size)
            case Load(vtype=vtype, ptr=ptr):
                return self._to_vtype(self._get_from_registers(ptr), vtype)
            case Icmp(vtype=vtype, op=op, lhs=lhs, rhs=rhs):
                logger.debug(f"icmp {vtype} {op} {lhs} {rhs}")
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._cmp_with_op(op, lhs, rhs)
            case Srem(vtype=vtype, lhs=lhs, rhs=rhs):
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._rem(lhs, rhs)
            case Add(vtype=vtype, lhs=lhs, rhs=rhs):
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._add(lhs, rhs)
            case Mul(vtype=vtype, lhs=lhs, rhs=rhs):
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._mul(lhs, rhs)
            case Call(name=name, args=args):
                logger.debug(f"calling {name} {args}")
                match name:
                    case "printf":
                        logger.debug(f"printf {args}")
                        args = [self.get_value(arg.value, arg.vtype) for arg in args]

                        def _maybe_to_char(value):
                            try:
                                return chr(int(value))
                            except ValueError:
                                return value

                        if self._program is not None and self._program.convert_numbers_to_chars:
                            args = [_maybe_to_char(arg) for arg in args]
                        logger.debug(f"printf {args}")
                        fstring = args[0]
                        # we need to replace all the percent-formatters (like %d) with {}
                        fstring = re.sub(r"%\w", "{}", fstring)
                        out_str = fstring.format(*args[1:])
                        print(out_str, end="")
                        return 0
                    case "srand":
                        seed = self.get_value(args[0].value, args[0].vtype)
                        self._rng = random.Random(seed)
                        return
                    case "scanf":
                        logger.debug(f"scanf {args}")
                        dst = args[1].value
                        prompt = self.get_value(args[0].value, "str")
                        assert prompt.strip() == "%c"
                        input_str = input()
                        char = ord(input_str)
                        self.store(dst, char)
                        return 0
                    case "rand":
                        return self._rng.randint(0, 100)
                raise NotImplementedError(name)
            case GetElementPtr(vtype=vtype, ptr=ptr, idx=idx):
                logger.debug(f"getting element ptr from {ptr} -> {idx}")
                idx = self.get_value(idx)
                logger.debug(f"idx into array is {idx}")
                return f"arr ptr={ptr} idx={idx}"

        raise NotImplementedError(expr)

    def _store_in_registers(self, reg, value):
        self.registers[reg] = value

    def _store(self, ptr, value):
        self._store_in_registers(ptr, value)

    def _is_array_ref(self, value):
        return isinstance(value, str) and value.startswith("arr ptr")

    def _parse_array_ref(self, value):
        m = re.match(r"arr ptr=(\S+) idx=(\S+)", value)
        ptr, idx = m.groups()
        return ptr, idx

    def _store_array(self, ptr, idx, value):
        logger.debug(f"called store_array ptr={ptr} idx={idx} value={value}")
        arr = self._get_from_registers(ptr)
        logger.debug(f"before store: arr = {arr}")
        arr[idx] = value
        logger.debug(f"after store: arr = {arr}")

    def assign(self, reg, value):
        logger.debug(f"assign {value} to {reg}")
        self._store(reg, value)

    def store(self, ptr, value):
        logger.debug(f"store {value} to {ptr}")
        # check if ptr points to an array reference
        reg_value = self._get_from_registers(ptr)
        logger.debug(f"ptr {ptr} exists, value = {reg_value}")
        if self._truthy(self._is_array_ref(reg_value)):
            logger.debug(f"ptr {ptr} is an array ref")
            arr_ptr, idx = self._parse_array_ref(reg_value)
            logger.debug(f"parsed array ref: {arr_ptr} {idx}")
            self._store_array(arr_ptr, idx, value)
            return
        self._store(ptr, value)

    def _truthy(self, value):
        return bool(value)

    def run_program(self, program: Program):
        self._program = program
        iptr = 0
        num_instr = 0

        while iptr < len(program.instructions):
            instr = program.instructions[iptr]
            logger.debug(f"{num_instr} || executing {instr}")
            num_instr += 1

            match instr:
                case Return(vtype=vtype, value=value):
                    logger.debug(f"return {value}")
                    return self.get_value(value, vtype=vtype)
                case Branch(label=label):
                    logger.debug(f"branch to {label}")
                    iptr = program.labels[label]
                    continue
                case BranchCond(
                    cond_reg=cond_reg, label_true=label_true, label_false=label_false
                ):
                    cond = self.get_value(cond_reg)
                    logger.debug(f"cond {cond_reg} = {cond} (type {type(cond)}))")
                    if self._truthy(cond):
                        logger.debug(f"branch to {label_true}")
                        iptr = program.labels[label_true]
                    else:
                        logger.debug(f"branch to {label_false}")
                        iptr = program.labels[label_false]
                    continue
                case Store(value=value, ptr=ptr):
                    logger.debug(f"store {value} to {ptr}")
                    value = self.get_value(value)
                    self.store(ptr, value)
                case Assign(reg=reg, expr=expr):
                    value = self.eval_expr(expr)
                    self.assign(reg, value)
                case Switch(ptr=ptr, default_label=default_label, cases=cases):
                    value = self.get_value(ptr)
                    logger.debug(f"switch on {value}")
                    for case, label in cases.items():
                        if self._truthy(self._cmp_with_op("eq", value, case)):
                            logger.debug(f"branch to case {case} -> {label}")
                            iptr = program.labels[cases[case]]
                            break
                    else:
                        logger.debug(f"branch to default {default_label}")
                        iptr = program.labels[default_label]
                    continue
                case _:
                    if isinstance(instr, Expr):
                        self.eval_expr(instr)
                    else:
                        raise NotImplementedError(instr)

            iptr += 1


openai.api_key = os.getenv("OPENAI_KEY")


class GPTVM(VM):
    name: str = "GPT VM"
    registers: dict[str, Any] = {}
    default_system: str = ""
    postfix: str = """Write your answer on a single line. Be very short and concise.
    I just want the plain answer, not the explanation."""

    def _gpt_get_answer(self, prompt, system=""):
        logger.debug(f"Getting answer for prompt: {prompt} | system: {system}")
        messages = []
        system = system or self.default_system
        if system:
            messages.append({"role": "system", "content": system})

        prompt = f"""{prompt}
        {self.postfix}"""

        messages.append({"role": "user", "content": prompt})
        for i in range(5):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=1024,
                    timeout=10,
                )
                break
            except openai.error.TryAgain:
                logger.debug(f"Try again {i}...")
                continue
        else:
            raise RuntimeError("Failed to get response from GPT")
        response = response["choices"][0]["message"]["content"].strip()  # type: ignore
        logger.debug(f"Got answer: {response}")
        return response



    def _to_vtype(self, value, vtype):
        return value

    get_from_registers_prompt : str = """Look at registers: {registers}.
    Now get me the value of {value}."""

    def _get_from_registers(self, value):
        return self._gpt_get_answer(
            self.get_from_registers_prompt.format(registers=self.registers, value=value)
        )

    add_prompt : str = """Add {lhs} and {rhs}."""

    def _add(self, lhs, rhs):
        return self._gpt_get_answer(self.add_prompt.format(lhs=lhs, rhs=rhs))

    rem_prompt : str = """Get the remainder value of {lhs} divided by {rhs}."""

    def _rem(self, lhs, rhs):
        return self._gpt_get_answer(self.rem_prompt.format(lhs=lhs, rhs=rhs))

    multiply_prompt : str = """Multiply {lhs} and {rhs}."""

    def _mul(self, lhs, rhs):
        return self._gpt_get_answer(self.multiply_prompt.format(lhs=lhs, rhs=rhs))

    load_array_prompt : str = """Get the value of index {idx} in array {arr}."""

    def _load_array(self, arr, idx):
        return self._gpt_get_answer(self.load_array_prompt.format(arr=arr, idx=idx))

    def _truthy(self, value):
        if isinstance(value, bool):
            b = value
        elif isinstance(value, int):
            b = bool(value)
        elif isinstance(value, str):
            if any(t in value.lower() for t in ("yes", "true")):
                b = True
            else:
                b = False
        else:
            raise NotImplementedError(value)
        logger.debug(f"truthiness of {value} = {b}")
        return b

    cmp_eq_prompt : str = """Are {lhs} and {rhs} equal?"""
    cmp_ne_prompt : str = """Are {lhs} and {rhs} different?"""
    cmp_sgt_prompt : str = """Is {lhs} greater than {rhs}?"""
    cmp_sge_prompt : str = """Is {lhs} greater than or equal to {rhs}?"""
    cmp_slt_prompt : str = """Is {lhs} less than {rhs}?"""
    cmp_sle_prompt : str = """Is {lhs} less than or equal to {rhs}?"""
    cmp_system : str = """You are a very keen observer."""

    def _cmp_with_op(self, op, lhs, rhs) -> bool:
        match op:
            case "eq":
                prompt = self.cmp_eq_prompt.format(lhs=lhs, rhs=rhs)
            case "ne":
                prompt = self.cmp_ne_prompt.format(lhs=lhs, rhs=rhs)
            case "sgt":
                prompt = self.cmp_sgt_prompt.format(lhs=lhs, rhs=rhs)
            case "sge":
                prompt = self.cmp_sge_prompt.format(lhs=lhs, rhs=rhs)
            case "slt":
                prompt = self.cmp_slt_prompt.format(lhs=lhs, rhs=rhs)
            case "sle":
                prompt = self.cmp_sle_prompt.format(lhs=lhs, rhs=rhs)
            case _:
                raise NotImplementedError(op)
        return self._gpt_get_answer(system=self.cmp_system, prompt=prompt)

    is_array_ref_prompt : str = """Is {value} an array reference?"""

    def _is_array_ref(self, value):
        return self._gpt_get_answer(self.is_array_ref_prompt.format(value=value))

    parse_array_ref_prompt : str = """Parse array reference {value}. Give me the array pointer and the index separated by a pipe symbol (|)"""

    def _parse_array_ref(self, value):
        return self._gpt_get_answer(self.parse_array_ref_prompt.format(value=value)).split(
            "|"
        )

    store_array_prompt : str = """Store {value} in array {arr} at index {idx}. Give me back the full new array after the update."""

    def _store_array(self, ptr, idx, value):
        arr = self._get_from_registers(ptr)
        answer = self._gpt_get_answer(
            self.store_array_prompt.format(value=value, arr=arr, idx=idx)
        )
        self._store_in_registers(ptr, answer)


class FullGPTVM(GPTVM):
    name: str = "Full GPT VM"
    registers: str = f"{({i: None for i in range(32)})}"

    store_in_registers_prompt : str = """Store the value "{value}" in register {reg}. Here are the current registers: {registers}. Give me back the complete state of the registers after the update."""

    def _store_in_registers(self, reg, value):
        self.registers = self._gpt_get_answer(
            self.store_in_registers_prompt.format(
                value=value, reg=reg, registers=self.registers
            )
        )


class ChadGPTVM(FullGPTVM):
    name: str = "Chad GPT VM"

    postfix : str = """Bro I know you love to talk, but I really need just the answer.
    Please don't explain your answer, and don't write pleaseantaries, like "the answer is...".
    Just give me the plain answer.
    Here's an example of a good answer: 42. That's it. Just the number. Limit prose. Be super concise."""

    get_from_registers_prompt : str = """Bro! Look at those rad registers: {registers}.
    Bro fr just tell me me the value of register {value}."""
    add_prompt : str = """Yo, broooo! Yo what's the sum of {lhs} and {rhs}?"""
    rem_prompt : str = """Bro fr you know how remainders and stuff works?
    I really need the remainder value of {lhs} divided by {rhs}. What is it bro?"""
    multiply_prompt : str = """My maaaaan, you mad good at multiplication, right?
    Yo what's {lhs} times {rhs}?"""
    load_array_prompt : str = """Bro listen! I have this array here: {arr}.
    And I just need the value of index {idx}. Can you help me out?"""
    cmp_eq_prompt : str = """Bro I'm so confused. Are {lhs} and {rhs} equal?"""
    cmp_ne_prompt : str = """Bro, my brain is melting. Are {lhs} and {rhs} different?"""
    cmp_sgt_prompt : str = """What's up bro? Tell me this: Is {lhs} greater than {rhs}?"""
    cmp_sge_prompt : str = """You know what's awesome? Comparing numbers.
    Tell me this: Is {lhs} greater than or equal to {rhs}?"""
    cmp_slt_prompt : str = """Bro I'm actually not sure about this one:
    Is {lhs} less than {rhs}?"""
    cmp_sle_prompt : str = """My main man! I need your help with this one:
    Is {lhs} less than or equal to {rhs}?"""
    cmp_system : str = """Bro!"""
    is_array_ref_prompt : str = """Bro look at this: "{value}". Is this an array reference that looks like "arr ptr=..."?
    Like does it look like it would point to an array?"""
    parse_array_ref_prompt : str = """Bro. I really need to parse this array reference: {value}.
    Can you give me the array reference and the index separated by a pipe symbol (|)?"""
    store_array_prompt : str = """Bro, look, I have this array here: {arr}.
    I need you to store the value "{value}" in it at index {idx}. Can you give me back the full new array after the update?"""
    store_in_registers_prompt : str= """My main maaan! I need you to store the value "{value}" in register "{reg}".
    Look here, these are the current registers: {registers}.
    Can you give me back the complete state of the registers after the update? Thanks bro!"""
