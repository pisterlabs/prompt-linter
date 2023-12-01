from pathlib import Path

import numpy as np
import pytest
import torch
from pretty_tools.datastruct import mdict
from pretty_tools import PATH_PRETTY
from pretty_tools.datastruct import bbox_convert_np as np_converter
from copy import deepcopy
from pretty_tools.test.core import (
    np_Max_error,
    error_fp128,
    error_fp64,
    error_fp32,
    error_fp16,
    error_fp8,
)

from pretty_tools.datastruct import cython_bbox

test_ltrb = np.array(
    [
        [827.82458496, 738.5569458, 937.07751465, 983.7911377],
        [1171.41674805, 51.3410759, 1214.76159668, 160.52073288],
        [1147.53527832, 693.87664795, 1269.04736328, 960.7043457],
        [320.17575073, 650.61608887, 434.10876465, 912.78771973],
        [1818.14257812, 270.06488037, 1915.07556152, 462.42419434],
        [508.56640625, 163.56619263, 564.137146, 303.78341675],
        [10.59531975, 216.00405884, 129.67011833, 379.37728882],
        [278.41213989, 198.63432312, 340.89767456, 345.30392456],
        [1021.55444336, 368.80096436, 1098.07055664, 577.02966309],
    ]
)
test_ann = np.array(
    [
        [1, 0, 1.8735e03, 3.6050e02, 6.5000e01, 1.7700e02, 1, 1, -1, -1],
        [1, 2, 1.2070e03, 8.2500e02, 1.0400e02, 2.6600e02, 1, 1, -1, -1],
        [1, 3, 3.7600e02, 7.7400e02, 1.1400e02, 2.5200e02, 1, 1, -1, -1],
        [1, 7, 1.1920e03, 1.0600e02, 4.0000e01, 1.0400e02, 1, 1, -1, -1],
        [1, 8, 6.4000e01, 2.9650e02, 1.0800e02, 1.7100e02, 1, 1, -1, -1],
        [1, 13, 8.7300e02, 8.5500e02, 1.0200e02, 2.4800e02, 1, 1, -1, -1],
        [1, 15, 5.3150e02, 2.4000e02, 6.3000e01, 1.3400e02, 1, 1, -1, -1],
        [1, 17, 1.0585e03, 4.6850e02, 7.5000e01, 2.0700e02, 1, 1, -1, -1],
        [1, 18, 3.0950e02, 2.6900e02, 6.1000e01, 1.4200e02, 1, 1, -1, -1],
        [1, 25, 2.7750e02, 1.1100e02, 4.7000e01, 9.4000e01, 1, 1, -1, -1],
    ]
)

test_ann_torch = torch.as_tensor(test_ann)


class Test_Bbox_Convert:
    def setup_method(self):
        self.bbox1_fp64 = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        self.bbox1_fp32 = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        self.bbox1_fp16 = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float16)

        self.bbox2_fp64 = np.array([[0.3, 0.4, 0.5, 0.6], [1.3, 1.4, 1.5, 1.6]], dtype=np.float64)
        self.bbox2_fp32 = np.array([[0.3, 0.4, 0.5, 0.6], [1.3, 1.4, 1.5, 1.6]], dtype=np.float32)
        self.bbox2_fp16 = np.array([[0.3, 0.4, 0.5, 0.6], [1.3, 1.4, 1.5, 1.6]], dtype=np.float16)

    def test_pynp_convert_1d(self):
        """
        一维矩阵
        测试python的numpy模块设计的转换


        """

        from pretty_tools.datastruct.bbox_convert_np import ltrb_to_ltwh, ltrb_to_xywh, ltwh_to_ltrb, ltwh_to_xywh, xywh_to_ltrb, xywh_to_ltwh

        check = np_Max_error
        #! 估计误差的时候，由于计算的时候存在一次加减运算，最后又要和一个存在误差的真值做差，因此误差累计是3倍
        assert check(ltrb_to_ltwh(self.bbox1_fp64), np.array([0.3, 0.4, 0.2, 0.2], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltrb_to_ltwh(self.bbox1_fp32), np.array([0.3, 0.4, 0.2, 0.2], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltrb_to_ltwh(self.bbox1_fp16), np.array([0.3, 0.4, 0.2, 0.2], dtype=np.float16)) <= 3 * error_fp16

        assert check(ltwh_to_xywh(self.bbox1_fp64), np.array([0.55, 0.7, 0.5, 0.6], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltwh_to_xywh(self.bbox1_fp32), np.array([0.55, 0.7, 0.5, 0.6], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltwh_to_xywh(self.bbox1_fp16), np.array([0.55, 0.7, 0.5, 0.6], dtype=np.float16)) <= 3 * error_fp16

        assert check(xywh_to_ltrb(self.bbox1_fp64), np.array([0.05, 0.1, 0.55, 0.7], dtype=np.float64)) <= 3 * error_fp64
        assert check(xywh_to_ltrb(self.bbox1_fp32), np.array([0.05, 0.1, 0.55, 0.7], dtype=np.float32)) <= 3 * error_fp32
        assert check(xywh_to_ltrb(self.bbox1_fp16), np.array([0.05, 0.1, 0.55, 0.7], dtype=np.float16)) <= 3 * error_fp16

        assert check(ltrb_to_xywh(self.bbox1_fp64), np.array([0.4, 0.5, 0.2, 0.2], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltrb_to_xywh(self.bbox1_fp32), np.array([0.4, 0.5, 0.2, 0.2], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltrb_to_xywh(self.bbox1_fp16), np.array([0.4, 0.5, 0.2, 0.2], dtype=np.float16)) <= 3 * error_fp16

        assert check(xywh_to_ltwh(self.bbox1_fp64), np.array([0.05, 0.1, 0.5, 0.6], dtype=np.float64)) <= 3 * error_fp64
        assert check(xywh_to_ltwh(self.bbox1_fp32), np.array([0.05, 0.1, 0.5, 0.6], dtype=np.float32)) <= 3 * error_fp32
        assert check(xywh_to_ltwh(self.bbox1_fp16), np.array([0.05, 0.1, 0.5, 0.6], dtype=np.float16)) <= 3 * error_fp16

        assert check(ltwh_to_ltrb(self.bbox1_fp64), np.array([0.3, 0.4, 0.8, 1.0], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltwh_to_ltrb(self.bbox1_fp32), np.array([0.3, 0.4, 0.8, 1.0], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltwh_to_ltrb(self.bbox1_fp16), np.array([0.3, 0.4, 0.8, 1.0], dtype=np.float16)) <= 3 * error_fp16

    def test_convert_2d(self):
        """
        二维矩阵
        测试python的numpy模块设计的转换


        """
        from pretty_tools.datastruct.bbox_convert_np import ltrb_to_ltwh, ltrb_to_xywh, ltwh_to_ltrb, ltwh_to_xywh, xywh_to_ltrb, xywh_to_ltwh

        check = np_Max_error

        assert (
            check(
                ltrb_to_ltwh(self.bbox2_fp64),
                np.array([[0.3, 0.4, 0.2, 0.2], [1.3, 1.4, 0.2, 0.2]], dtype=np.float64),
            )
            <= 3 * error_fp64
        )
        assert (
            check(
                ltrb_to_ltwh(self.bbox2_fp32),
                np.array([[0.3, 0.4, 0.2, 0.2], [1.3, 1.4, 0.2, 0.2]], dtype=np.float32),
            )
            <= 3 * error_fp32
        )
        assert (
            check(
                ltrb_to_ltwh(self.bbox2_fp16),
                np.array([[0.3, 0.4, 0.2, 0.2], [1.3, 1.4, 0.2, 0.2]], dtype=np.float16),
            )
            <= 3 * error_fp16
        )

        assert check(ltwh_to_xywh(self.bbox2_fp64), np.array([[0.55, 0.7, 0.5, 0.6], [2.05, 2.2, 1.5, 1.6]], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltwh_to_xywh(self.bbox2_fp32), np.array([[0.55, 0.7, 0.5, 0.6], [2.05, 2.2, 1.5, 1.6]], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltwh_to_xywh(self.bbox2_fp16), np.array([[0.55, 0.7, 0.5, 0.6], [2.05, 2.2, 1.5, 1.6]], dtype=np.float16)) <= 3 * error_fp16

        assert check(xywh_to_ltrb(self.bbox2_fp64), np.array([[0.05, 0.1, 0.55, 0.7], [0.55, 0.6, 2.05, 2.2]], dtype=np.float64)) <= 3 * error_fp64
        assert check(xywh_to_ltrb(self.bbox2_fp32), np.array([[0.05, 0.1, 0.55, 0.7], [0.55, 0.6, 2.05, 2.2]], dtype=np.float32)) <= 3 * error_fp32
        assert check(xywh_to_ltrb(self.bbox2_fp16), np.array([[0.05, 0.1, 0.55, 0.7], [0.55, 0.6, 2.05, 2.2]], dtype=np.float16)) <= 3 * error_fp16

        assert check(ltrb_to_xywh(self.bbox2_fp64), np.array([[0.4, 0.5, 0.2, 0.2], [1.4, 1.5, 0.2, 0.2]], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltrb_to_xywh(self.bbox2_fp32), np.array([[0.4, 0.5, 0.2, 0.2], [1.4, 1.5, 0.2, 0.2]], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltrb_to_xywh(self.bbox2_fp16), np.array([[0.4, 0.5, 0.2, 0.2], [1.4, 1.5, 0.2, 0.2]], dtype=np.float16)) <= 3 * error_fp16

        assert check(xywh_to_ltwh(self.bbox2_fp64), np.array([[0.05, 0.1, 0.5, 0.6], [0.55, 0.6, 1.5, 1.6]], dtype=np.float64)) <= 3 * error_fp64
        assert check(xywh_to_ltwh(self.bbox2_fp32), np.array([[0.05, 0.1, 0.5, 0.6], [0.55, 0.6, 1.5, 1.6]], dtype=np.float32)) <= 3 * error_fp32
        assert check(xywh_to_ltwh(self.bbox2_fp16), np.array([[0.05, 0.1, 0.5, 0.6], [0.55, 0.6, 1.5, 1.6]], dtype=np.float16)) <= 3 * error_fp16

        assert check(ltwh_to_ltrb(self.bbox2_fp64), np.array([[0.3, 0.4, 0.8, 1.0], [1.3, 1.4, 2.8, 3.0]], dtype=np.float64)) <= 3 * error_fp64
        assert check(ltwh_to_ltrb(self.bbox2_fp32), np.array([[0.3, 0.4, 0.8, 1.0], [1.3, 1.4, 2.8, 3.0]], dtype=np.float32)) <= 3 * error_fp32
        assert check(ltwh_to_ltrb(self.bbox2_fp16), np.array([[0.3, 0.4, 0.8, 1.0], [1.3, 1.4, 2.8, 3.0]], dtype=np.float16)) <= 3 * error_fp16

        pass


class Test_MDICT:
    def setup_method(self):
        pass

    def test_mdict_set(self):
        x = mdict()  # 创建一个二重索引字典（默认）
        a = mdict(3)  # 创建一个三重索引字典
        b = mdict(5)  # 创建一个五重索引字典

        a[1, 2, "3"] = "example"
        a["1", 4, "3"] = "example"
        a[0, 0, 0] = 0
        b[1, 1, 1, 1, 1] = "五重索引字典"
        b[0, 1, 2, 3, 4] = 0

        # * 检查赋值是否正确
        assert a[0, 0, 0] == 0
        assert a["1", 4, "3"] == "example"
        assert a[1, 2, "3"] == "example"

        assert a[1, 2, "3"] == a["1", 4, "3"]
        assert a[0, 0, 0] != a[1, 2, "3"]

        # * 检查多重索引是否正确
        assert a[1, 2, "3"] == a[1, "3", 2]
        assert b[1, 2, 3, 4, 0] == b[0, 1, 2, 3, 4]
        assert id(b[1, 1, 1, 1, 1]) == id(b[1, 1, 1, 1, 1])
        assert id(b[1, 2, 3, 4, 0]) == id(b[0, 1, 2, 3, 4])

        # * 检查 in 魔法方法是否正确
        assert [1, 2, "3"] in a
        assert ["3", 2, 1] in a
        assert ["x", "x", "y"] not in a

    def test_mdict_combinations_list(self):
        list_data = []
        a = np.arange(2, 5)
        b = np.arange(3, 6)
        c = np.arange(4, 7)
        list_data = [a, b, c]
        # * ----------------------------------------
        # * 测试顺序无关
        fn = lambda a, b: a + b
        m_result = mdict.combinations(list_data, fn)
        pass
        assert (m_result[0, 1] == np.array([5, 7, 9])).all(), f"m_result[0, 1] as {m_result[0, 1]} != [5, 7, 9]"
        assert (m_result[1, 0] == np.array([5, 7, 9])).all(), f"m_result[1, 0] as {m_result[1, 0]} != [5, 7, 9]"
        # * ----------------------------------------
        # * 测试顺序相关
        fn = lambda a, b: np.outer(a, b)
        m_result = mdict.combinations(list_data, fn)
        assert id(m_result[0, 1]) == id(m_result[1, 0])
        assert (m_result[0, 1] == np.outer(a, b)).all(), "测试结果受顺序影响的排列组合生成器，确保顺序是 a*b，因为a在b的前面"
        assert (m_result[0, 1] != np.outer(b, a)).any(), f"{m_result[0, 1]} 与 {np.outer(b, a)} 应当不同，测试结果受顺序影响的排列组合生成器，确保顺序是 a*b，因为a在b的前面"

    def test_mdict_combinations_dict(self):
        # * ----------------------------------------
        # * 测试输入为字典的生成器
        a = np.arange(2, 5)
        b = np.arange(3, 6)
        c = np.arange(4, 7)
        dict_data = {"a": a, "b": b, "c": c}
        fn = lambda a, b: np.outer(a, b)
        m_result = mdict.combinations(dict_data, fn)
        assert (m_result["a", "b"] == np.outer(a, b)).all(), "测试结果受顺序影响的排列组合生成器，确保顺序是 a*b，因为a在b的前面"
        assert (m_result["b", "a"] == np.outer(a, b)).all(), "测试结果受顺序影响的排列组合生成器，确保顺序是 a*b，因为a在b的前面"
        assert (m_result["a", "b"] != np.outer(b, a)).any(), f"{m_result['a', 'b']} 与 {np.outer(b, a)} 应当不同，测试结果受顺序影响的排列组合生成器，确保顺序是 a*b，因为a在b的前面"

    def test_mdict_apply_fn(self):
        a = np.arange(2, 5)
        b = np.arange(3, 6)
        c = np.arange(4, 7)
        dict_data = {"a": a, "b": b, "c": c}
        fn = lambda a, b: a + b
        m_result = mdict.combinations(dict_data, fn)
        fn = lambda x: 2 * x
        m_result = mdict.apply(m_result, fn)
        assert (m_result["a", "b"] == np.array([10, 14, 18])).all()
        assert (m_result["b", "c"] == np.array([14, 18, 22])).all()
        assert (m_result["c", "a"] == np.array([12, 16, 20])).all()
        fn = lambda v, k: v + ord(k[0])
        m_result = mdict.apply(m_result, fn)  # * 测试一个函数，每个值加一下索引号第一位的ASCII码
        assert (m_result["a", "b"] == np.array([107, 111, 115])).all()
        assert (m_result["b", "c"] == np.array([112, 116, 120])).all()
        assert (m_result["c", "a"] == np.array([109, 113, 117])).all()

    def test_mdict_num_items(self):
        # * num_items 是输入所有参数的计数器
        pass
        x = mdict()  # 创建一个二重索引字典（默认）
        x[1, 2] = 0
        assert x.num_items == 2

        x[1, 2] = 1
        x[2, 3] = "a"
        x["k", 4] = 0
        assert x.num_items == 5
        del x[2, 3]  # 删掉了一个2 和一个3，但是2还是存留的，所以num_items减到4
        assert x.num_items == 4
        del x["k", 4]
        assert x.num_items == 2

    def test_mdict_merge_block(self):
        """
        输入一个二重索引字典，输出一个块矩阵
        """

        pass
        # todo


class Test_Numpy_Enhance:
    def test_bisect(self):
        from pretty_tools.datastruct.np_enhance import bisect_left, bisect_right

        ptr = np.array([0, 9, 15, 20])
        # ---------------------------- bisect_left ----------------------------
        assert bisect_left(ptr, 0) == 0  #! 区别于 bisect_right
        assert bisect_left(ptr, 1) == 1
        assert bisect_left(ptr, 9) == 1  #! 区别于 bisect_right
        assert bisect_left(ptr, 10) == 2
        assert bisect_left(ptr, 12) == 2
        assert bisect_left(ptr, 15) == 2  #! 区别于 bisect_right
        assert bisect_left(ptr, 16) == 3
        assert bisect_left(ptr, 20) == 3  #! 区别于 bisect_right

        # ---------------------------- bisect_right ----------------------------
        assert bisect_right(ptr, 0) == 1  #! 区别于 bisect_left
        assert bisect_right(ptr, 1) == 1
        assert bisect_right(ptr, 9) == 2  #! 区别于 bisect_left
        assert bisect_right(ptr, 10) == 2
        assert bisect_right(ptr, 12) == 2
        assert bisect_right(ptr, 15) == 3  #! 区别于 bisect_left
        assert bisect_right(ptr, 16) == 3
        assert bisect_right(ptr, 20) == 4  #! 区别于 bisect_left

        # ---------------------------- bisect_right_array ----------------------------
        assert (bisect_left(ptr, np.arange(20)) == np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3])).all()
        assert (bisect_right(ptr, np.arange(20)) == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])).all()

    def test_np_block(self):
        from pretty_tools.datastruct import np_enhance

        # * 随机生成一个 10*20大小的矩阵
        tmp_a = np.random.random((10, 20))
        # * 指定其数据划分方式为 [[5, 8], [7, 12, 16]], 因此应当被切分成 大小为 (3, 4)的矩阵
        index_slice = [[5, 8], [7, 12, 16]]

        np_block = np_enhance.block(tmp_a, index_slice)

        assert (np_block[0, 1] == tmp_a[:5, 7:12]).all()
        assert (np_block[2, 1] == tmp_a[8:, 7:12]).all()
        assert (np_block[1, 3] == tmp_a[5:8, 16:]).all()
        index_slice = [[0, 5, 8], [7, 12, 16, 20]]  # * 补足最后一个节点或者起始节点，结果应当不变
        np_block = np_enhance.block(tmp_a, index_slice)
        assert (np_block[0, 1] == tmp_a[:5, 7:12]).all()
        assert (np_block[2, 1] == tmp_a[8:, 7:12]).all()
        assert (np_block[1, 3] == tmp_a[5:8, 16:]).all()

        # * 判断三维切片
        tmp_a = np.random.random((10, 20, 30))
        index_slice = [[5, 8], [7, 12, 16], [10, 20]]
        np_block = np_enhance.block(tmp_a, index_slice)
        assert (np_block[0, 1, 1] == tmp_a[:5, 7:12, 10:20]).all()
        assert (np_block[2, 1, 2] == tmp_a[8:, 7:12, 20:]).all()
        assert (np_block[1, 3, 0] == tmp_a[5:8, 16:, :10]).all()

        # * 测试分块发生改变后，原始矩阵是否发生改变
        tmp = np_block[0, 1, 1]
        tmp += 1
        assert (np_block[0, 1, 1] == tmp_a[:5, 7:12, 10:20]).all()
        tmp = np_block[0, 0, 0]
        tmp[0, 0, 0] = -1000
        assert tmp_a[0, 0, 0] == -1000

    def test_from_index(self):
        import torch  # * 虽然引入了torch，但是这个类原本是应当能够
        from pretty_tools.datastruct import np_enhance

        np1 = np_enhance.from_index(
            torch.tensor(
                [
                    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9],
                    [3, 7, 1, 5, 7, 2, 5, 8, 4, 7, 6, 0, 8, 9, 6, 1, 2, 7, 8, 9, 4, 1, 3, 5, 6, 9, 4, 8, 6, 4],
                ]
            ),
            shape=(10, 10),
        )  # 长度为 10
        np2 = np_enhance.from_index(
            torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], [4, 5, 1, 3, 5, 2, 3, 1, 5, 1, 2, 5, 5, 0, 1, 1, 4, 3, 2, 3, 1]]),
            shape=(7, 7),
        )  # 长度为7
        np3 = np_enhance.from_index(
            torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], [6, 3, 2, 2, 3, 0, 3, 0, 6, 2, 0, 6, 6, 0, 3, 0, 6, 3, 0, 3, 2]]),
            shape=(7, 7),
        )  # 长度为 7
        pass
        assert (
            np1
            == np.array(
                [
                    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                ]
            )
        ).all()
        assert (
            np2
            == np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert (
            np3
            == np.array(
                [
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test_np_block_merge(self):
        from pretty_tools.datastruct import np_enhance

        dict_edge = {}
        dict_edge[0] = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            ]
        )  # 长度为 10
        dict_edge[1] = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )  # 长度为7
        dict_edge[2] = np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )  # 长度为 7

        np_block = np_enhance.block.merge_from(dict_edge.values(), [(i, i) for i in range(len(dict_edge))])
        assert np_Max_error(np_block[0, 0], dict_edge[0]) == 0
        assert np_Max_error(np_block[1, 1], dict_edge[1]) == 0
        assert np_Max_error(np_block[2, 2], dict_edge[2]) == 0

    def test_remap(self):
        pass
        from pretty_tools.datastruct import np_enhance

        np_matched_id = np.array([0, 2, 3, 17, 2, 3, 13, 17, 2, 3, 17, 21, 33])
        x = np_enhance.remap(np_matched_id)
        assert (
            np_Max_error(
                x,
                np.array([0, 1 / 6, 2 / 6, 4 / 6, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 1 / 6, 2 / 6, 4 / 6, 5 / 6, 1.0]),
            )
            <= error_fp64
        )
        pass

    def test_index_value_2d(self) -> None:
        from pretty_tools.datastruct import np_enhance

        np_test = np.arange(4, dtype=float).reshape(2, 2)
        result1 = np_enhance.index_value_2d(np_test)
        result2 = np_enhance.index_value_2d(np_test, nonzero=True)
        assert len(result1) == 4
        assert (result1 == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 3.0]])).all()
        assert len(result2) == 3
        assert (result2 == np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 1.0, 3.0]])).all()
        pass

    def test_index_convert_to_block(self):
        from pretty_tools.datastruct import np_enhance

        index_slice = [
            [0, 10, 17, 24],
            [0, 10, 17, 24],
        ]
        list_index = [
            [0, 3, 7, 8, 9, 9, 14, 15, 16],
            [21, 18, 14, 10, 12, 17, 20, 21, 22],
        ]
        index_block, index_inner = np_enhance.index_convert_to_block(index_slice, list_index)
        assert (
            index_block
            == np.array(
                [
                    [0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [2, 2, 1, 1, 1, 2, 2, 2, 2],
                ]
            )
        ).all()
        assert (
            index_inner
            == np.array(
                [
                    [0, 3, 7, 8, 9, 9, 4, 5, 6],
                    [4, 1, 4, 0, 2, 0, 3, 4, 5],
                ]
            )
        ).all()
        pass

    def test_index_convert_to_combine(self):
        from pretty_tools.datastruct import np_enhance

        index_slice = [[0, 10, 17, 24], [0, 10, 17, 24]]
        index_inner = [[0, 3, 7, 8, 9, 9, 4, 5, 6], [4, 1, 4, 0, 2, 0, 3, 4, 5]]
        index_block = [[0, 0, 0, 0, 0, 0, 1, 1, 1], [2, 2, 1, 1, 1, 2, 2, 2, 2]]
        index_combine = np_enhance.index_convert_to_combine(index_slice, index_block, index_inner)
        assert (index_combine == np.array([[0, 3, 7, 8, 9, 9, 14, 15, 16], [21, 18, 14, 10, 12, 17, 20, 21, 22]])).all()


class Test_Numpy_Bbox:
    def setup_method(self):
        pass
        self.np_bboxes = np.array(
            [
                [0, 0, 4, 5],
                [2, 2, 8, 8],
                [2, 2, 9, 9],
                [10, 0, 15, 10],
                [0, 0, 3, 7],
            ],
            dtype=np.float64,
        )

    def test_iou_overlap_flag(self):
        from pretty_tools.datastruct.cython_bbox import cy_bbox_overlaps_flag

        overlap_flag = cy_bbox_overlaps_flag(self.np_bboxes, self.np_bboxes)
        gt = np.array(
            [
                [True, True, True, False, True],
                [True, True, True, False, True],
                [True, True, True, False, True],
                [False, False, False, True, False],
                [True, True, True, False, True],
            ]
        )
        assert (overlap_flag == gt).all()

    def test_iou_no_overlap(self):
        from pretty_tools.datastruct.numpy_bbox import bbox_no_overlaps_area

        # 测试计算违被遮挡面积
        result = bbox_no_overlaps_area(self.np_bboxes, self.np_bboxes)
        assert (result == np.array([2.0, 0.0, 13.0, 50.0, 4.0])).all()

        # 测试计算未被遮挡面积占比
        result = bbox_no_overlaps_area(self.np_bboxes, self.np_bboxes, ratio=True)
        assert (result == np.array([2.0 / 20, 0.0, 13.0 / 49, 1, 4.0 / 21])).all()


class Test_GeneralAnn:
    def setup_method(self):
        pass

    def test_dataframe_change(self):
        from pandas import DataFrame

        data1 = DataFrame([["1", "2", 3.5, 500, 9.9], ["a", "b", 4.9, 1000, 9999.9999]])
        data1.columns = ["str1", "str2", "num1", "num2", "float_int"]
        assert data1["num1"].dtype == np.float64
        assert data1["num2"].dtype == int

        data1.loc[:, "num1"] = [-3.5, -4.9]  # * 这里会使得数据发生改变
        # * 下面两种判断方式结果都一样
        assert (data1["num1"].to_numpy() == np.array([-3.5, -4.9])).all()
        assert (data1["num1"] == [-3.5, -4.9]).all()

        # * 直接索引赋值
        data1["num2"] = [
            -500,
            -1000,
        ]  # ? 这里会使得数据发生改变, 但是被提示不安全，因为不能确定直接索引返回的是一个视图还是一个浅拷贝

        #! 下面的变更数据类型方法不会起效果，需要注意
        assert data1.loc[:, "float_int"].dtype == np.float64
        assert (data1.loc[:, "float_int"] == [9.9, 9999.9999]).all()
        data1.loc[:, "float_int"] = data1["float_int"].astype(int)
        assert data1.loc[:, "float_int"].dtype == np.float64  #! 类型并没有发生改变
        assert not (data1.loc[:, "float_int"] == [9.9, 9999.9999]).all()  #! 但是值发生了改变, 这里执行的是赋值
        assert (data1.loc[:, "float_int"] == [9.0, 9999.0]).all()  #! 但是值发生了改变

    def test_reformat_bbox(self):
        """
        测试自动格式化 shape == 4 的数据
        """
        from pretty_tools.datastruct.general_ann import reformat_ann_dataframe

        ann_ltrb = reformat_ann_dataframe(test_ltrb)
        # * 检查数据类型
        assert ann_ltrb["prob"].dtype == np.float64
        assert ann_ltrb["frame"].dtype == int
        assert ann_ltrb["id"].dtype == int
        assert ann_ltrb["cls"].dtype == int
        assert ann_ltrb["crowd"].dtype == int

        # * 用 传入四个参数的二维数组生成一个 DataFrame对象，没有标明数据来源的锚框格式时，默认为xywh
        assert (np.array(ann_ltrb)[:, 0] == -1).all()  # * 只有4个参数，则帧号为-1
        assert (np.array(ann_ltrb)[:, 1] == -1).all()  # * 只有4个参数，则id为-1
        assert (ann_ltrb["frame"] == -1).all()
        assert (ann_ltrb["id"] == -1).all()
        assert (ann_ltrb[["xc", "yc", "w", "h"]] == test_ltrb).all().all()  # * 没有发生格式转换，所以这里的数据没有发生变化

        # * 检查显式 锚框格式转换
        ann_xywh = reformat_ann_dataframe(test_ltrb, str_format="ltrb")
        assert (np.array(ann_xywh)[:, 0] == -1).all()  # * 只有4个参数，则帧号为-1
        assert (np.array(ann_xywh)[:, 1] == -1).all()  # * 只有4个参数，则id为-1
        assert (ann_xywh["frame"] == -1).all()
        assert (ann_xywh["id"] == -1).all()
        assert (ann_xywh[["xc", "yc", "w", "h"]] == np_converter.ltrb_to_xywh(test_ltrb)).all().all()  # * 没有发生格式转换，所以这里的数据没有发生变化

    def test_reformat_ann(self):
        """
        测试自动格式化 shape >= 6 的数据
        """
        from pretty_tools.datastruct.general_ann import reformat_ann_dataframe

        ann_xywh = reformat_ann_dataframe(test_ann, str_format="xywh")
        (ann_xywh["id"] == test_ann[:, 1]).all()
        # todo 测试还没写全，但是懒得写了

    def test_generalbboxes_setter(self):
        from pretty_tools.datastruct import GeneralAnn

        bboxes = GeneralAnn(test_ann, str_format="xywh")

        check_id = id(bboxes.ltrb[:, 2])
        bboxes.ltrb[:, 2] = 2  #! 这种索引方式修改的是缓存中的 数组，对真实的数据没有任何影响
        bboxes.get_ltrbs.cache_clear()

        tmp = np.random.random((bboxes.shape[0], 4))

        # * 检查 set 方法
        bboxes.set_ltrb(tmp)
        assert np_Max_error(tmp, bboxes.ltrb) < error_fp64

        bboxes.set_ltwh(tmp)
        assert np_Max_error(tmp, bboxes.ltwh) < error_fp64

        bboxes.set_xywh(tmp)
        assert np_Max_error(tmp, bboxes.xywh) < error_fp64

    def test_generalbboxes_cache(self):
        from pretty_tools.datastruct import GeneralAnn

        bboxes = GeneralAnn(test_ltrb, str_format="ltrb")

        bboxes = GeneralAnn(test_ann, str_format="xywh")
        assert id(bboxes.ltrb) == id(bboxes.ltrb)  # * 判断缓存机制是否生效
        assert id(bboxes.xywh) == id(bboxes.xywh)  # * 判断缓存机制是否生效
        assert id(bboxes.ltwh) == id(bboxes.ltwh)  # * 判断缓存机制是否生效
        assert id(bboxes.ids) == id(bboxes.ids)  # * 判断缓存机制是否生效
        assert id(bboxes.frames) == id(bboxes.frames)  # * 判断缓存机制是否生效
        # todo 输入格式为 List[np.ndarray] 还没测试过，以及之后的torch.Tensor格式也还没补全

    def test_autonorm(self):
        """测试自动归一化功能"""
        from pretty_tools.datastruct import GeneralAnn

        origin_wh = (1920, 1080)
        norm_ann = deepcopy(test_ann)
        norm_ann[:, 2:6] /= [*origin_wh, *origin_wh]
        # 初始归一化，测试 renorm 为 False
        bboxes_ann = GeneralAnn(ori_ann=test_ann, str_format="xywh", ori_WH=origin_wh)
        assert np_Max_error(bboxes_ann.xywh, test_ann[:, 2:6] / [*origin_wh, *origin_wh]) < error_fp64
        bboxes_ann.set_ori_WH((3840, 2160), renorm=False)  # * 设定了新的尺度大小，但是没有要求重新归一化，则归一化锚框尺寸不变，但是反归一化后的尺寸会发生改变，即扩大两倍
        assert np_Max_error(bboxes_ann.ori_xywh, 2 * test_ann[:, 2:6]) < error_fp64  # 相差两倍校验
        assert np_Max_error(bboxes_ann.xywh, norm_ann[:, 2:6]) < error_fp64

        # 初始归一化，测试 renorm 为 True（默认）
        bboxes_ann = GeneralAnn(ori_ann=test_ann, str_format="xywh", ori_WH=origin_wh)
        bboxes_ann.set_ori_WH((3840, 2160))  # * 设定了新的更大的尺度大小，并要求重新归一化，先用旧的尺寸恢复到原来真实水平，然后根据新的尺寸进行归一化，实际上归一化的标注缩小了
        assert np_Max_error(bboxes_ann.ori_xywh, test_ann[:, 2:6]) < error_fp64  # 反归一化应当不变
        assert np_Max_error(bboxes_ann.xywh, norm_ann[:, 2:6] / 2) < error_fp64  # 归一化尺寸被迫缩小两倍

        # 实例化时没有输入尺寸，后续手动输入，测试是否自动归一化
        bboxes_ann = GeneralAnn(ori_ann=test_ann, str_format="xywh")
        bboxes_ann.set_ori_WH((3840, 2160))  # 用一个二倍大的尺寸去归一化，实际上归一化的标注缩小了
        assert np_Max_error(bboxes_ann.ori_xywh, test_ann[:, 2:6]) < error_fp64
        assert np_Max_error(bboxes_ann.xywh, norm_ann[:, 2:6] / 2) < error_fp64

        # 使用已经被归一化的数据实例化对象，测试是否能够正常还原，实例化时输入尺寸
        bboxes_ann = GeneralAnn(ann=norm_ann, str_format="xywh", ori_WH=origin_wh)
        assert np_Max_error(bboxes_ann.ori_xywh, test_ann[:, 2:6]) < error_fp64
        assert np_Max_error(bboxes_ann.xywh, norm_ann[:, 2:6]) < error_fp64

        # 使用已经被归一化的数据实例化对象，测试是否能够正常还原，实例化后输入尺寸
        bboxes_ann = GeneralAnn(ann=norm_ann, str_format="xywh")
        bboxes_ann.set_ori_WH((1920, 1080))
        assert np_Max_error(bboxes_ann.ori_xywh, test_ann[:, 2:6]) < error_fp64  # 反归一化应当不变
        assert np_Max_error(bboxes_ann.xywh, norm_ann[:, 2:6]) < error_fp64  # 归一化尺寸被迫缩小两倍

    def test_init_with_image(self):
        pass
        from pretty_tools.datastruct import GeneralAnn
        from PIL import Image

        image = Image.open(PATH_PRETTY.joinpath("resources/imgs/Circle_View1_000001.jpg"))
        bboxes = GeneralAnn(ori_ann=test_ann, str_format="xywh", ori_img=image)
        assert bboxes.ori_img == image
        assert bboxes.ori_WH == image.size

        # * 初始化的时候传入一个强制的尺度，这个尺度会覆盖原始图像的尺度
        bboxes = GeneralAnn(ori_ann=test_ann, str_format="xywh", ori_img=image, ori_WH=(2000, 1000))
        assert bboxes.ori_WH == (2000, 1000)


class Test_TrackInstance:
    def setup_method(self):
        import cv2
        from PIL import Image

        self.test_img = Image.open(PATH_PRETTY.joinpath("resources/imgs/Circle_View1_000001.jpg"))

    def test_build(self):
        from pretty_tools.datastruct import TrackCameraInstances

        test_trackinstances = TrackCameraInstances(test_ann, str_format="xywh")

        assert id(test_trackinstances.ltrb) == id(test_trackinstances.ltrb)  # * 判断缓存机制是否生效
        assert id(test_trackinstances.xywh) == id(test_trackinstances.xywh)  # * 判断缓存机制是否生效
        assert id(test_trackinstances.ltwh) == id(test_trackinstances.ltwh)  # * 判断缓存机制是否生效

    def test_build_from_ann(self):
        from pretty_tools.datastruct import GeneralAnn, TrackCameraInstances

        test_generalann = GeneralAnn(ori_ann=test_ltrb, str_format="ltrb")
        test_trackinstances = TrackCameraInstances(from_general=test_generalann)
        assert np_Max_error(test_trackinstances.xywh, test_generalann.xywh) < error_fp64
        assert np_Max_error(test_trackinstances.ltrb, test_generalann.ltrb) < error_fp64
        assert np_Max_error(test_trackinstances.ltwh, test_generalann.ltwh) < error_fp64

        #! 检查是否自动归一化
        test_trackinstance = TrackCameraInstances(
            from_general=test_generalann,
        )
        test_trackinstance.set_ori_WH((1920, 1080), renorm=True)

        assert test_trackinstance.xywh[:, :2].max() < 2.0
        assert test_trackinstance.xywh[:, :2].min() > -1.0

    def test_build_with_embbeding(self):
        from pretty_tools.datastruct import TrackCameraInstances

        test_trackinstances = TrackCameraInstances(test_ann, str_format="xywh")
        test_trackinstances.embeddings = {}
        test_feat = np.load(PATH_PRETTY.joinpath("resources/data/test_feat_9x2048.npy"))

        test_trackinstances.embeddings["curr_feat"] = test_feat
        pass


class Test_TrackGraph:
    def setup_method(self):
        # fmt: off
        self.edgelist = np.array([[0, 2], [0, 3], [0, 8], [1, 8], [1, 5], [1, 4],
                                  [2, 0], [2, 8], [2, 3], [3, 0], [3, 7], [3, 6],
                                  [4, 1], [4, 8], [4, 2], [5, 7], [5, 6], [5, 8],
                                  [6, 7], [6, 5], [6, 3], [7, 5], [7, 6],
                                  [7, 3], [8, 2], [8, 1], [8, 0]])
        # fmt: on
        self.test_feat = np.load(PATH_PRETTY.joinpath("resources/data/test_feat_9x2048.npy"))

    def test_build_by_ltrb(self):
        from pretty_tools.datastruct.track_graph import TrackCameraGraph

        #! 方法1
        test_trackgraph = TrackCameraGraph(
            ann=test_ltrb,
            str_format="ltrb",
            node_x=self.test_feat,
            edge_index=self.edgelist,
        )

        #! 方法2
        test_trackgraph = TrackCameraGraph(
            ori_ann=test_ltrb,
            str_format="ltrb",
        )
        test_trackgraph.set_node_x(self.test_feat)
        test_trackgraph.set_edge_index(self.edgelist)
        test_trackgraph.validate(raise_on_error=True)

    def test_build_from_ann(self):
        from pretty_tools.datastruct import GeneralAnn, TrackCameraGraph

        #! 方法1
        test_generalann = GeneralAnn(ori_ann=test_ltrb, str_format="ltrb")
        test_trackgraph = TrackCameraGraph(
            from_general=test_generalann,
            node_x=self.test_feat,
            edge_index=self.edgelist,
        )
        test_trackgraph.validate(raise_on_error=True)

        #! 方法3
        test_trackgraph = TrackCameraGraph(
            from_general=test_generalann,
        )
        test_trackgraph.set_node_x(self.test_feat)
        test_trackgraph.set_edge_index(self.edgelist)
        test_trackgraph.validate(raise_on_error=True)

        #! 检查是否自动归一化
        test_trackgraph = TrackCameraGraph(from_general=test_generalann)
        test_trackgraph.set_ori_WH((1920, 1080), renorm=True)
        assert test_trackgraph.xywh[:, :2].max() < 2.0
        assert test_trackgraph.xywh[:, :2].min() > -1.0

    def test_convert(self):
        """
        测试两个类型的数据是否可以相互转换
        """
        from pretty_tools.datastruct import (
            GeneralAnn,
            TrackCameraGraph,
            TrackCameraInstances,
        )

        test_generalann = GeneralAnn(test_ltrb, str_format="ltrb")

        # * 测试 TrackCameraInstances 转换到 TrackCameraGraph
        test_trackinstance = TrackCameraInstances(from_general=test_generalann)
        test_instance_to_graph = TrackCameraGraph(from_general=test_trackinstance)
        pass

        # * 测试从 TrackCameraGraph 转换到 TrackCameraInstances
        test_trackgraph = TrackCameraGraph(from_general=test_generalann)
        test_graph_to_instance = TrackCameraInstances(from_general=test_trackgraph)
        pass

    def test_convert_with_image(self):
        """
        测试两个类型的数据是否可以相互转换
        """
        from pretty_tools.datastruct import (
            GeneralAnn,
            TrackCameraGraph,
            TrackCameraInstances,
        )
        from PIL import Image

        image = Image.open(PATH_PRETTY.joinpath("resources/imgs/Circle_View1_000001.jpg"))
        test_generalann = GeneralAnn(test_ltrb, str_format="ltrb", ori_img=image)

        # * 测试 TrackCameraInstances 转换到 TrackCameraGraph
        test_trackinstance = TrackCameraInstances(from_general=test_generalann)
        test_instance_to_graph = TrackCameraGraph(from_general=test_trackinstance)
        assert test_trackinstance.ori_img == image
        assert test_instance_to_graph.ori_img == image
        assert test_trackinstance.ori_WH == image.size
        assert test_instance_to_graph.ori_WH == image.size

        # * 测试从 TrackCameraGraph 转换到 TrackCameraInstances
        test_trackgraph = TrackCameraGraph(from_general=test_generalann)
        test_graph_to_instance = TrackCameraInstances(from_general=test_trackgraph)
        assert test_trackgraph.ori_img == image
        assert test_graph_to_instance.ori_img == image
        assert test_trackgraph.ori_WH == image.size
        assert test_graph_to_instance.ori_WH == image.size

    def test_save_and_load(self):
        # * 即时保存
        from pretty_tools.datastruct import (
            GeneralAnn,
            TrackCameraGraph,
            TrackCameraInstances,
        )

        test_trackgraph = TrackCameraGraph(
            ori_ann=test_ltrb,
            str_format="ltrb",
            node_x=self.test_feat,
            edge_index=self.edgelist,
        )
        TrackCameraGraph.save(test_trackgraph, PATH_PRETTY.joinpath("/tmp/test_generalann.pkl"))

        test_load_trackgraph = TrackCameraGraph.load(PATH_PRETTY.joinpath("/tmp/test_generalann.pkl"))
        assert (test_load_trackgraph.ltrb == test_trackgraph.ltrb).all(), "校验两个的信息是否一致"

        pass
        # * 读取预设的
        test_load_trackgraph = TrackCameraGraph.load(PATH_PRETTY.joinpath("resources/data/test_generalann.pkl"))
        assert (test_load_trackgraph.ltrb == test_trackgraph.ltrb).all(), "校验两个的信息是否一致"


class Test_Graph_Enhance:
    def setup_method(self):
        from pretty_tools.resources import PATH_RESOURCES_DATA
        from pretty_tools.datastruct.graph_enhance import CohereGraph

        self.x = np.loadtxt(PATH_RESOURCES_DATA.joinpath("test_feat_20x2048.txt"), delimiter=",")
        self.list_len = [4, 6, 10]

        self.cograph = CohereGraph(self.x, list_len=self.list_len)
        pass

    def test_onehot(self):
        """
        这里其实测的是 :class:`np_enhance` 中的 `cython` 函数
        """
        assert (
            self.cograph.x_onehot
            == np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        ).all()
        pass


if __name__ == "__main__":
    pytest.main(
        [
            "-s",
            "-l",
            "test_datastruct.py",
        ]
    )
