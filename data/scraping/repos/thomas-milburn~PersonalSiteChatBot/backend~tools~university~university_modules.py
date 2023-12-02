import json
from typing import Optional, Type, List, Dict, Any

from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

with open("context/university_modules.json", "r") as f:
    all_modules_brief = [
        {
            "code": module["code"],  # e.g. "COM1001
            "name": module["name"],
            "university_year": module["university_year"],
            "grade_achieved_percent": module["grade_percent"],
            "more_info": module["short_link"],
        }
        for module in json.load(f)
    ]

with open("context/university_modules.json", "r") as f:
    all_modules_details = json.load(f)


class GetUniversityModulesArgs(BaseModel):
    university_year: Optional[int] = Field(
        description="The year of university to get modules for. If not provided, all modules will be returned",
        default=None,
    )


class GetUniversityModules(BaseTool):
    name = "get_university_modules"
    description = """Returns the module codes, names, university year, grade achieved, more info link of the modules Thomas carried out at university"""
    args_schema: Type[GetUniversityModulesArgs] = GetUniversityModulesArgs

    def _run(
        self,
        university_year: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Use the tool."""
        if university_year is not None:
            return [
                module
                for module in all_modules_brief
                if module["university_year"] == university_year
            ]

        return all_modules_brief

    async def _arun(
        self,
        university_year: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Use the tool asynchronously."""
        if university_year is not None:
            return [
                module
                for module in all_modules_brief
                if module["university_year"] == university_year
            ]

        return all_modules_brief


class UniversityModuleDescriptionArgs(BaseModel):
    module_code: str = Field(
        description="The module code to get the description of"
    )


class GetUniversityModuleDescription(BaseTool):
    name = "get_university_modules_description"
    description = """Returns the description of the module with the given module id"""
    args_schema: Type[UniversityModuleDescriptionArgs] = UniversityModuleDescriptionArgs

    def _run(
        self,
        module_code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Use the tool."""
        for module in all_modules_brief:
            if module["code"].lower() == module_code.lower():
                return {
                    "code": module["code"],  # e.g. "COM1001
                    "name": module["name"],
                    "description": module["description"],
                }

        return {"error": "Module not found"}

    async def _arun(
        self,
        module_code: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Use the tool asynchronously."""
        for module in all_modules_details:
            if module["code"].lower() == module_code.lower():
                return {
                    "code": module["code"],  # e.g. "COM1001
                    "name": module["name"],
                    "description": module["description"],
                }

        return {"error": "Module not found"}
