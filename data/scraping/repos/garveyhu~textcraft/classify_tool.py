from langchain.tools import BaseTool

from textcraft.tools.label_tool import LabelTool


class ClassifyTool(BaseTool):
    name = "分类工具"
    description = "分类工具"
    labels = ""

    def _run(self, text: str, run_manager=None) -> str:
        return self.run_for_classify(text)

    async def _arun(
        self,
        text: str,
        run_manager=None,
    ) -> str:
        pass

    def run_for_classify(self, text):
        print(self.labels)
        return LabelTool().get_label(text, self.labels)


if __name__ == "__main__":
    classify = ClassifyTool()
    sentence = "某股份制银行推出的1年期、2年期、3年期的礼仪存单，利率分别是2.25%、2.85%、3.50%，每个期限的礼仪存单利率只比同期限的大额存单利率低0.05个百分点。"
    label = classify.run_for_classify(sentence)
    print(label)
