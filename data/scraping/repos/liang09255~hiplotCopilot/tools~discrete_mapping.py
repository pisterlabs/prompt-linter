import uuid
import pandas
from langchain.tools import BaseTool
from common.is_number import is_number


class DiscreteMapping(BaseTool):
    name: str = "discrete_mapping"
    description: str = """
    Input to this tool is an excel file path with optional manual mapping dict and manual mapping col.
    If the tool runs successfully, the output will be the path of the excel file with mapping result end of *_discrete_mapping_*.xlsx.
    Example Input: data/thing.xlsx
    Params: file_path(required), manual_mapping_dict(optional), manual_mapping_col(optional)
    Set manual_mapping_dict can map values manually, at the same time you should set manual_mapping_col to None.
    Set manual_mapping_col can map specify columns, at the same time you should set manual_mapping_dict to None.
    """

    outputPath: str = "data/tmp/"

    def __init__(self):
        super().__init__()
        self.return_direct = True

    def _run(self, file_path: str, manual_mapping_dict: dict = None, manual_mapping_col: list = None) -> str:
        # 读取文件
        try:
            data = pandas.read_excel(file_path)
        except FileNotFoundError:
            return ""
        row = len(data)
        if row == 0:
            return ""
        col = len(data.iloc[0])

        # 暂存映射关系(映射关系表)
        storage_mapping_dict = []

        for i in range(col):
            # 跳过纯数字列
            if is_number(data.iloc[0, i]):
                continue
            # 手动指定列时跳过未指定列
            if manual_mapping_col is not None and data.columns[i] not in manual_mapping_col:
                continue
            # 遍历离散值所在列进行赋值
            mapping_dict = {}
            for j in range(row):
                # 手动指定映射
                if manual_mapping_dict is not None:
                    if data.iloc[j, i] in manual_mapping_dict:
                        data.iloc[j, i] = manual_mapping_dict[data.iloc[j, i]]
                    continue
                # 自动映射
                if data.iloc[j, i] not in mapping_dict:
                    mapping_dict[data.iloc[j, i]] = len(mapping_dict) + 1
                data.iloc[j, i] = mapping_dict[data.iloc[j, i]]
            # 保存映射关系
            storage_mapping_dict.append([data.columns[i], ""])
            # key value
            for k, v in mapping_dict.items():
                storage_mapping_dict.append([k, v])
            storage_mapping_dict.append(["", ""])

        # 保存
        file_name = file_path.split("/")[-1]
        short_file_name = file_name.split(".")[0]
        file_id = uuid.uuid4()
        destination_path = self.outputPath + short_file_name + "_discrete_mapping_" + str(file_id) + ".xlsx"

        # 写数据表
        data.to_excel(destination_path, index=False)

        # 追加映射关系
        mapping_dict = pandas.DataFrame(storage_mapping_dict)
        with pandas.ExcelWriter(destination_path, mode='a', engine='openpyxl') as writer:
            mapping_dict.to_excel(writer, index=False, header=False, sheet_name="mapping")
        return destination_path


if __name__ == '__main__':
    dis = DiscreteMapping()
    dis.outputPath = "../data/tmp/"
    data_file_path = "../data/thing.xlsx"
    # data_file_path = "../data/baby.xls"
    print(dis.run({
        "file_path": data_file_path,
        "manual_mapping_dict": {
            "蓝绿": 123,
        }
    }))
    # dis.run({
    #     "file_path": data_file_path,
    # })
    print(dis.run({
        "file_path": data_file_path,
        "manual_mapping_col": ["颜色", "类型"],
    }))
