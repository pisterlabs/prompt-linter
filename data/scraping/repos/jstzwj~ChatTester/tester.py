# Copyright 2023 by XiaHan. All rights reserved.
# This file is part of the ChatTester,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import json
import os
from typing import List, Optional, Tuple
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from chattester.source import UnitTestPair
from chattester.checker import UnitTestChecker
from chattester.maven_parser import MavenOutputParser

def parse_java_code_from_answer(answer: str) -> Optional[str]:
    idx = answer.find("```java")
    if idx == -1:
        return None
    else:
        end_idx = answer.rfind("```")
        java_code = answer[idx + 7: end_idx]
        return java_code


JUNIT_IMPORT = """
import static org.junit.Assert.fail;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertSame;

import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.lang.reflect.Type;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;
"""

class ChatGPTUnitTestGenerator(object):
    def __init__(self, project_path: str, openai_api_key: str, openai_api_base: Optional[str] = None) -> None:
        self.project_path = project_path
        self.model_name = "gpt-3.5-turbo"
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base

        self.mvn_parser = MavenOutputParser()

        self.model = ChatOpenAI(
            model_name=self.model_name, 
            temperature=0.9, 
            max_tokens=2048,
            openai_api_key=self.openai_api_key,
            openai_api_base=self.openai_api_base,
        )
        self.memory = ConversationBufferMemory()
        self.llm_chain = ConversationChain(
            llm=self.model,
            memory=self.memory
        )
        self.role = "You are a professional who writes Java test methods."
        self.basic_prompt = PromptTemplate(
            input_variables=["focal", "role_instruction", "focal_method_name", "junit_version"],
            template="""```java
{focal}
```
{role_instruction}
Please write a test method for the "{focal_method_name}" based on the given information using {junit_version}.""",
        )

        self.intention_prompt = PromptTemplate(
            input_variables=["focal", "focal_method_name"],
            template="""```java
{focal}
```
Please infer the intention of the "{focal_method_name}".""")
        
        self.generation_prompt = PromptTemplate(
            input_variables=["intention", "role_instruction", "focal_method_name", "junit_version"],
            template="""
// Method intention
{intention}
{role_instruction}
Please write a test method for the "{focal_method_name}" with the given Method intention using {junit_version}.""")
        
        self.bug_prompt_with_context = PromptTemplate(
            input_variables=["test_method", "test_context", "class_name"],
            template="""
// Test Method
{test_method}
{test_context}
The test method has a bug error(marked <Buggy Line>).
Please repair the buggy line with the given "{class_name}" class information and return the complete test method after repair.
Note that the "{class_name}" class information cannot be modified.""")
        
        self.bug_prompt = PromptTemplate(
            input_variables=["test_method"],
            template="""
```java
// Test Method
{test_method}
```
The test method has a bug error(marked <Buggy Line>).
Please repair the buggy line and return the complete test method after repair.""")

    def _get_focal_part(self, pair: UnitTestPair):
        fields = []
        for field in pair.focal_class.fields:
            fields.append(field.text)
        fields_str = "\n".join(fields)
        methods = []
        for method in pair.focal_class.methods:
            if method.name == pair.focal_method.name:
                continue
            methods.append(" ".join(method.modifiers) + " " + method.declaration + ";")
        methods_str = "\n".join(methods)
        template = f"""
// Focal Class
public class {pair.focal_class.name} {{
    {fields_str}
    {methods_str}
    // Focal method
    {pair.focal_method.text}
}}
"""
        return template
    
    def _mark_buggy_line(self, unittest: str, marks: List[Tuple[int, str]]):
        marked_unittest = unittest.splitlines(keepends=False)

        marked_unittest_with_line = []
        for i, unit in enumerate(marked_unittest):
            line = i + 1
            marked_unittest_with_line.append((line, unit))
        
        insert_records = []
        for mark_num, mark_line in marks:
            for line_num, line in marked_unittest_with_line:
                if mark_num == line_num:
                    insert_records.append((line_num, mark_line))
                    break
        
        for line_num, mark_line in sorted(insert_records, key=lambda x:x[0], reverse=True):
            marked_unittest_with_line.insert(line_num - 1, (-1, mark_line))
        
        return "\n".join([l[1] for l in marked_unittest_with_line])


    def basic_generate(self, pair: UnitTestPair):
        self.memory.clear()
        focal_str = self._get_focal_part(pair)
        query = self.basic_prompt.format(
            focal=focal_str,
            role_instruction=self.role,
            focal_method_name=pair.focal_method.declaration,
            junit_version="Junit4",
        )
        # with open("out.txt", "w", encoding="utf-8") as f:
        #     f.write(query)
        answer = self.llm_chain.run(query)
        # with open("answer.txt", "w", encoding="utf-8") as f:
        #     f.write(answer)

        focal_imports = [i for i in pair.imports if "java" not in i]
        focal_imports = "\n".join(focal_imports)

        return f"""{pair.package_info}
{JUNIT_IMPORT}
{focal_imports}
{parse_java_code_from_answer(answer)}
"""
    
    def _complete_unittest(self, test: str, core_class_name: str, package_info: str, focal_imports: str) -> str:
        if "public class" in test:
            return f"{package_info}\n{focal_imports}\n{test}"
        else:
            return f"{package_info}\n{focal_imports}\npublic class {core_class_name} {{\n{test}\n}}"

    def iterative_generate(self, pair: UnitTestPair, max_n: int=2):
        chat_history = []
        checker = UnitTestChecker(self.project_path)

        self.memory.clear()

        # intention query
        focal_str = self._get_focal_part(pair)
        intention_query = self.intention_prompt.format(
            focal=focal_str,
            focal_method_name=pair.focal_method.declaration,
        )
        intention_answer = self.llm_chain.run(intention_query)

        chat_history.append((intention_query, intention_answer))

        # generation query
        generation_query = self.generation_prompt.format(
            intention=intention_answer,
            role_instruction=self.role,
            focal_method_name=pair.focal_method.declaration,
            junit_version="Junit4",
        )

        generation_answer = self.llm_chain.run(generation_query)
        chat_history.append((generation_query, generation_answer))

        focal_imports = [i for i in pair.imports if "java" not in i]
        focal_imports = "\n".join(focal_imports)

        file_path = os.path.normpath(pair.test_path).replace("\\", "/")
        core_class_name = file_path.split("/")[-1].replace(".java", "")

        unittest = self._complete_unittest(
            parse_java_code_from_answer(generation_answer),
            core_class_name,
            pair.package_info,
            focal_imports,
        )

        new_test_file_path = pair.test_path.replace(".java", "Generation.java")

        gen_test = unittest
        for i in range(max_n):
            gen_test = gen_test.replace(
                    checker.get_core_class_name(pair.test_path),
                    checker.get_core_class_name(new_test_file_path)
            )
            checker.create_test(new_test_file_path, gen_test)
            output, success = checker.run_tests()
            # checker.remove_test(new_test_file_path)

            parsed_output = self.mvn_parser.parse(output)
            if parsed_output.status == "success":
                break
            error_output = parsed_output.filter("error")

            marks = []
            for each_error in error_output:
                error_file = each_error.get_path()
                if error_file is None:
                    continue
                
                if not error_file.endswith("Generation.java"):
                    continue

                line, col = each_error.get_line_col()
                buggy_msg = each_error.get_message().replace('\n', '\n// ')
                marks.append((line, f"// <Buggy Line>: {buggy_msg}"))
            
            buggy_marked_test = self._mark_buggy_line(gen_test, marks)

            self.memory.clear()
            # buggy query
            buggy_query = self.bug_prompt.format(
                test_method=buggy_marked_test,
            )

            buggy_answer = self.llm_chain.run(buggy_query)
            chat_history.append((buggy_query, buggy_answer))

            gen_test = self._complete_unittest(
                parse_java_code_from_answer(buggy_answer),
                core_class_name,
                pair.package_info,
                focal_imports,
            )

        with open("out.txt", "w", encoding="utf-8") as f:
            for line in chat_history:
                f.write(line[0] + "\n")
                f.write("--------------\n")
                f.write(line[1] + "\n")
                f.write("==============\n")
        return gen_test


