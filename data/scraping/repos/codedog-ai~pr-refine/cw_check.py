from __future__ import annotations

import asyncio
from typing import Optional

from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from pydantic import BaseModel
from streamlit.runtime.uploaded_file_manager import UploadedFile

from codedog_sdk.chains.bool_chain import BoolChain
from codedog_sdk.documents.base import Chapter, DocumentIndex
from codedog_sdk.documents.feishu_docx_splitter import FeishuDocxSplitter
from pr_refine.utils import load_gpt35_llm

OBJECTIVE_NOT_FOUND = "需求文档中不包含项目目标"
OBJECTIVE_NOT_CLEAR = "项目目标不够落地、清晰、可执行"
OBJECTIVE_OK = "项目目标清晰"
OBJ_LIST_NOT_FOUND = "需求文档中不包含项目范围"
OBJ_LIST_OK = "项目范围清晰"
REQUIREMENT_NOT_FOUND = "需求文档中不包含详细需求"
REQUIREMENT_NOT_ENOUGH = "需求数量较少"


class Report(BaseModel):
    table: list[dict[str, str | int | float | bool]]
    flag: bool
    cost: float = 0.0
    report: str
    name: str


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
        else:
            raise


class Checker:
    template_objective_parent = """帮我分析一篇《项目需求计划书》，以下是其中一个章节的目录，请帮我判定该章节中是否包含项目的整体目标.
目录:
{index}"""
    template_objective = """一篇《项目需求计划书》中有一个章节叫做{name},从名称推断，该章节是否包含项目的整体目标呢？"""
    template_obj_list = """一篇《项目需求计划书》中有一个章节叫做{name},从名称推断，该章节是否包含项目的需求范围呢？"""
    template_req = """帮我分析一篇《项目需求计划书》，以下是其中一个章节的目录，请帮我判定该章节内的每一节是否是一条一条的具体的项目需求.
目录:
{index}"""
    template_eval_obj = """帮我分析一篇《项目需求计划书》的项目目标计划内容是否是具体、明确，可落地，可执行的需求，而非抽象模糊的概念性质的目标
项目目标内容：
{content}
"""

    def __init__(self, file: UploadedFile):
        self.name = file.name
        self.doci = FeishuDocxSplitter().generate_doci(file)
        self._objective_parent_chain: BoolChain = BoolChain.from_llm(
            load_gpt35_llm(), self.template_objective_parent
        )
        self._objective_chain: BoolChain = BoolChain.from_llm(
            load_gpt35_llm(), self.template_objective
        )
        self._obj_list_chain: BoolChain = BoolChain.from_llm(
            load_gpt35_llm(), self.template_obj_list
        )
        self._req_chain: BoolChain = BoolChain.from_llm(
            load_gpt35_llm(), self.template_req
        )
        self._eval_obj_chain: BoolChain = BoolChain.from_llm(
            load_gpt35_llm(), self.template_eval_obj
        )

    def check(self) -> Report:
        with get_openai_callback() as cb:
            future1 = self.check_objective(cb)
            future2 = self.check_requirement(cb)

            loop = get_or_create_eventloop()
            future = loop.run_until_complete(asyncio.gather(future1, future2))

            cost = cb.total_cost

        f1, obj_report = future[0]
        f2, req_report = future[1]

        flag = f1 and f2
        report = f"1.{obj_report}\n2.{req_report}"
        return Report(
            table=[{"目录": chapter.pretty_name()} for chapter in self.doci.chapters],
            flag=flag,
            cost=cost,
            report=report,
            name=self.name,
        )

    async def check_objective(self, cb: BaseCallbackHandler) -> tuple[bool, str]:
        objective_chapter = await self._parse_objective(self.doci, cb)
        if not objective_chapter:
            return False, OBJECTIVE_NOT_FOUND

        objective = self.doci.get_by_chapter(objective_chapter)
        flag = await self._eval_objective_quality(objective)

        if not flag:
            return False, OBJECTIVE_NOT_CLEAR

        parent_chapter = objective_chapter.parent
        if not parent_chapter:
            return False, OBJ_LIST_NOT_FOUND
        obj_list_chapter = await self._parse_objective_list(parent_chapter, cb)
        if not obj_list_chapter:
            return False, OBJ_LIST_NOT_FOUND
        return True, OBJECTIVE_OK

    async def check_requirement(self, cb: BaseCallbackHandler) -> tuple[bool, str]:
        req_chapter: Optional[Chapter] = await self._parse_reqs(self.doci, cb)
        if not req_chapter:
            return False, REQUIREMENT_NOT_FOUND

        if len(req_chapter.children) < 3:
            return False, REQUIREMENT_NOT_ENOUGH

        return True, "包含详细需求"

    async def _parse_objective(
        self, doci: DocumentIndex, cb: BaseCallbackHandler
    ) -> Chapter | None:
        for i, root_chapter in enumerate(doci.root_chapters[:3]):  # 项目目标应当在前3章
            flag = await self._objective_parent_chain.arun(
                index=root_chapter.pretty_index(), callbacks=[cb]
            )
            if not flag:
                continue

            for sub_chapter in root_chapter.children:
                flag = await self._objective_chain.arun(
                    name=sub_chapter.name, callbacks=[cb]
                )
                if flag:
                    return sub_chapter

        return None

    async def _parse_objective_list(
        self, chapter: Chapter, cb: BaseCallbackHandler
    ) -> Chapter | None:
        for sub_chapter in chapter.children:
            flag = await self._obj_list_chain.arun(
                name=sub_chapter.name, callbacks=[cb]
            )
            if flag:
                return sub_chapter
        return None

    async def _parse_reqs(
        self, doci: DocumentIndex, cb: BaseCallbackHandler
    ) -> Chapter | None:
        for chapter in doci.root_chapters[1:]:  # 具体需求不在第一章
            flag = await self._req_chain.arun(
                index=chapter.pretty_index(), callbacks=[cb]
            )
            if flag:
                return chapter

    async def _eval_objective_quality(self, objective: Document) -> bool:
        flag = await self._eval_obj_chain.arun(content=objective.page_content)
        return flag
