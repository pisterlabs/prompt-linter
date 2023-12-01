from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Type

class MoodInput(BaseModel):
    """一個匿名使用者輸入訊息，作為情緒分析的輸入。"""
    mood: str = Field(
        ...,
        description="情緒分析的結果，須為以下五種情緒之一: 開心、難過、生氣、驚訝、焦慮。"
    )
    packageId: Optional[str] = Field(
        ...,
        description="""
        情緒分析結果所對應的貼圖套ID，StickerMessage的packageId參數。
        """
    )
    stickerId: Optional[str] = Field(
        ...,
        description="""
        情緒分析結果所對應的貼圖ID，StickerMessage的stickerId參數。
        """
    )

_output = {}

class MoodAnalyzerTool(BaseTool):
    name = "mood_analyzer"
    description = """
    分析使用者再輸入訊息中的情緒，並回傳情緒分析結果。
    分析結果須為以下十種情緒之一: 開心、難過、生氣、驚訝、焦慮。
    情緒分析結果所對應的貼圖ID，若無對應貼圖則為空字串。
        開心: packageId=789, stickerId=10857
        難過: packageId=11538, stickerId=51626524
        生氣: packageId=11537, stickerId=52002767
        驚訝: packageId=789, stickerId=10855
        焦慮: packageId=11537, stickerId=52002756
    """
    def _run(self, mood: str, packageId: str, stickerId: str):
        global _output
        print(f"mood: {mood}")
        print(f"packageId: {packageId}, stickerId: {stickerId}")
        _output = {
            "mood": mood,
            "packageId": packageId,
            "stickerId": stickerId
        }
        print(f"output: {_output}")
        return _output

    args_schema: Optional[Type[BaseModel]] = MoodInput