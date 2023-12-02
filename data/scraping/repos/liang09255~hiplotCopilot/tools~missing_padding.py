import uuid
import pandas
from langchain.tools import BaseTool
from common.is_number import is_number


class MissingPadding(BaseTool):
    name: str = "Missing padding"
    description: str = """
        Input to this tool is an excel file path with padding_type and optional padding_type.
        If the tool runs successfully, the output will be the path of the excel file with mapping result end of *_missing_padding_*.xlsx.
        Example Input: data/iris.xlsx
        Params: file_path(required), padding_col(optional), padding_type(required)
        Padding_type: 1 for zero padding, 2 for average padding.
        Padding_col: specify columns, if not set, all number columns will be padded.
        """
    outputPath: str = "data/tmp/"
    paddingType: tuple = (1, 2)

    def _run(self, file_path: str, padding_type: int, padding_col: list = None) -> str:
        # 参数校验
        if padding_type not in self.paddingType:
            return ""

        # 读取文件
        try:
            data = pandas.read_excel(file_path)
            if len(data) == 0:
                return ""
        except FileNotFoundError:
            return ""

        if padding_col is None or len(padding_col) == 0:
            padding_col = []
            # 找出所有数字列
            for i in range(len(data.iloc[0])):
                if is_number(data.iloc[0, i]):
                    padding_col.append(data.columns[i])

        # 处理
        if padding_type == 1:
            fill_zero(data, padding_col)
        elif padding_type == 2:
            fill_average(data, padding_col)

        # 保存
        file_name = file_path.split("/")[-1]
        short_file_name = file_name.split(".")[0]
        file_id = uuid.uuid4()
        destination_path = self.outputPath + short_file_name + "_missing_padding_" + str(file_id) + ".xlsx"

        data.to_excel(destination_path, index=False)

        return destination_path


# 填充零值
def fill_zero(data: pandas.DataFrame, padding_col: list):
    row_num = len(data)
    for col_name in padding_col:
        if col_name not in data.columns:
            continue
        for i in range(row_num):
            if pandas.isnull(data.iloc[i][col_name]):
                data.loc[i, col_name] = 0


# 填充平均值
def fill_average(data: pandas.DataFrame, padding_col: list):
    row_num = len(data)
    for col_name in padding_col:
        if col_name not in data.columns:
            continue
        sum = 0
        count = 0
        for i in range(row_num):
            if pandas.isnull(data.iloc[i][col_name]):
                continue
            sum += data.iloc[i][col_name]
            count += 1
        average = sum / count
        for i in range(row_num):
            if pandas.isnull(data.iloc[i][col_name]):
                data.loc[i, col_name] = average


if __name__ == '__main__':
    data_file_path = "../data/iris.xlsx"
    mis = MissingPadding()
    mis.outputPath = "../data/tmp/"
    print(mis.run({
        "file_path": data_file_path,
        "padding_col": ["sepal_width"],
        "padding_type": 2,
    }))
