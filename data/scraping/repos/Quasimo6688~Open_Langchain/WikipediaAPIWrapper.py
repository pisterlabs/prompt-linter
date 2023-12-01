from langchain.utilities.wikipedia import WikipediaAPIWrapper

# 输入和输出变量
input_query = None  # 用于存储从主程序或代理接收的查询
output_summaries = None  # 用于存储搜索结果的摘要，以便主程序或代理调用

# 搜索函数
def search(query):
    global output_summaries
    wrapper = WikipediaAPIWrapper()
    output_summaries = wrapper.run(query)

    return output_summaries
# 主程序
if __name__ == "__main__":
    input_query = "福特野马电动版跑车"  # 这里模拟从主程序或代理接收到的查询
    search(input_query)
    print("搜索结果摘要：", output_summaries)
