from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        try:
            from modelscope.pipelines import pipeline
        except ImportError:
            raise ImportError(
                "Could not import modelscope python package. "
                "Please install modelscope with `pip install modelscope`. "
            )


        p = pipeline(
            task="document-segmentation",
            model='damo/nlp_bert_document-segmentation_chinese-base',
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list

if __name__ == "__main__":
    text_splitter = AliTextSplitter(
        pdf = False
    )
    ls = [
        """
        标题
        全球可再生能源趋势

        简介
        近年来，全球能源格局正在发生重大变革。随着对气候变化和化石燃料有限性的担忧不断增长，世界正在将焦点转向可再生能源。这份简要报告旨在提供当前全球可再生能源趋势的概述。

        关键点

        太阳能迅猛增长： 太阳能在过去十年中取得了显著增长。成本下降，太阳能电池板效率提高，政府激励措施都促进了这一增长。

        风能扩张： 风能是另一个有前景的领域。离岸风电场越来越普及，风力涡轮机变得更加高效和具有成本效益。

        能源储存解决方案： 鉴于可再生能源如太阳能和风能的不确定性，能源储存解决方案，如先进的电池，对于电网的稳定性和可靠性至关重要。

        新兴技术： 在潮汐和地热能源等领域的研究和开发正在为清洁能源发电开辟新的可能性。

        政府政策： 许多国家的政府正在实施促进可再生能源的政策，包括补贴、税收激励措施和减排目标。

        挑战

        间歇性： 太阳能和风能等可再生能源的不可预测性为持续供能带来了挑战。

        基础设施投资： 转向可再生能源需要大量的基础设施投资，包括电网升级和新的能源储存设施。

        公众认知： 说服公众可再生能源的益处和可行性至关重要。

        结论
        全球转向可再生能源是在应对气候变化方面的一个令人鼓舞的趋势。然而，在将可再生能源整合到现有能源基础设施方面仍然存在挑战。持续的研究、投资和公众支持对于实现可持续能源未来至关重要。
        """,
        ]
    
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        num = 1
        for chunk in chunks:
            print("chunk_num_", num, end=" ")
            print(chunk)
            num += 1
