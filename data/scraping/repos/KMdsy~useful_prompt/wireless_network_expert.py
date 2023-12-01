import networkx as nx
import Levenshtein
import openai, pickle
from openai.api_resources.abstract import APIResource
# 输出每个案例中的根因解释: 
'''
prompt1: 异常解释
1. 解释系统预测出了什么异常，在什么位置
2. 以下是算法可能的根因列表「按照可能性的大小排序」
    1. 在：与此距离「多少距离」的基站上「某个指标」上发生的异常，可能预示着「距离过远、信道质量差...」
3. 请给出异常推理、传播过程，并用恰当的语言做出的解释

请以此格式输出：
异常首先发生在「」。
1. 然后传播至「」，导致「」。
2. 然后传播至「」，导致「」
...

prompt2: 异常链路提取
以下列表包含了一个指标的传播路径，请按照异常发生的顺序，将异常链路罗列为一个列表

请以此格式输出：
[KPI1, KPI2, ...]
'''
openai.api_base = 'https://api.chatanywhere.com.cn/v1'
openai.api_key = 'sk-tDEHE54fLU3MwYO7Gm2x5cOMGMYU3BmnWDXLCKLUDg6Tqivx'

def get_response(role=None, prompt=None):
    messages = []
    if role is not None:
        messages.append({'role': 'system', 'content': role})
    messages.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0301', messages=messages)
    output = response['choices'][0]['message']['content'].strip()
    return output

# 引导QOS理解
prefix_qos_explanation = '1. Explanation of QoS for candidate root causes:'
prefix_nl_explanation = '2. Analysis of anomaly propagation **STEP BY STEP**, then explain the propagation process in natural language: '
prefix_chain = '3. List the anomaly propagation steps based on the above analysis:'
suffix_qos_explanation = prefix_nl_explanation
suffix_nl_explanation = prefix_chain
suffix_chain = None

class RootCauseAnalysisPropmt:
    def __init__(self):
        # ref: https://chat.openai.com/share/6f8c2762-edb8-4b31-97a7-e7379885169c
        # translate: https://chat.openai.com/share/494428e4-04e6-4879-9f8c-b7549c943731
        self.output_format = f'''Please answer the following quesrtions and format your analysis as follows:
{prefix_qos_explanation}
  - Root cause 1: [Explain the quality of signal/channel/allocated resources/etc, — good/average/poor — when this indicator is at the given quantitative level]
  - [Add more explanations for root causes as per the actual situation]
{prefix_nl_explanation}
[Explain the propagation process in natural language]
{prefix_chain}
  - The anomaly first occurred at: [Indicator1]
  - Then propagated to: [Indicator2]
  - Then propagated to: [Indicator3]
  - Then propagated to: [Indicator4]
  - [Add more analysis here]
  - Eventually leading the anomaly in: [Indicator]

Output instructions: Please fill in your analysis results within the “[]” and maintain the content outside of the "[]" unchanged.'''
        self.role = 'You are a wireless network maintenance expert. Please help me analyze issues in the wireless access network. Based on the following information about **anomaly** and **candidate root causes**, please analyze the chain of anomalous propagation, that is, how root cause anomaly impacts other indicators, eventually leading to anomalies in the user experience indicators, and then use natural language to describe the process.\n'
        self.role += self.output_format

        self.n_level = 10

    def getPrompt(self, pred, causes):
        pred_pos, pred_rsrp, pred_kpi, pred_qos = pred.split('>>')
        causes_pos, causes_rsrp, causes_kpi, causes_qos = [], [], [], []
        for cause in causes:
            pos, rsrp, kpi, qos = cause.split('>>')
            causes_pos.append(pos)
            causes_rsrp.append(rsrp)
            causes_kpi.append(kpi)
            causes_qos.append(qos)

        output = 'Note: Not all candidate root causes are correct. You should use your professional knowledge to select the most likely root causes **from provided candidates** and explain their interrelationships.\n**An indicator name appears at most once**\n\n'
        output += f'1. Detected anomaly:\n  - Grid coordinates of the base station: ({pred_pos.split("_")[0]},{pred_pos.split("_")[1]})\n  - Anomaly indicators: {pred_kpi}\n  - Quantitative level of the indicators: {pred_qos}\n\n'
        output += '2. Candidate root causes, listed in the order in which they occurred:\n'
        for i, (pos, kpi, qos) in enumerate(zip(causes_pos, causes_kpi, causes_qos)):
            output += f'  {i+1}. Candidate {i+1}\n'
            output += f'    - Grid coordinates of the base station: ({pos.split("_")[0]},{pos.split("_")[1]})\n'
            output += f'    - Anomalous indicators observed: {kpi}\n'
            output += f'    - Quantitative level of the indicators: {qos}\n'
        output += f'\nIn the information provided, indicators are quantified into {self.n_level} levels, with level 1 indicating low values and level {self.n_level} indicating high values. You need to deduce the relationship between the indicator levels and QoS levels based on the meaning of the indicators and your professional knowledge.\n\n'
        
        return output
    
    def getResult(self, pred, causes):
        prompt = self.getPrompt(pred, causes)
        result = get_response(self.role, prompt)
        return result
    
class GrammarPrompt:
    def __init__(self) -> None:
        self.output_format = '''Please format your output as follows:

Score: [Fill in the score]
Reasons: [Provide reasons for scoring from the perspectives of grammar and professionalism]

Output instructions: Please fill in your analysis results within the "[]", and keep the content outside the "[]" unchanged.'''
        self.role = '''You are an expert in the field of wireless communications. I will provide a piece of analytical text about the process of anomaly propagation, describing how an anomaly in one indicator might spread to other indicators, ultimately leading to abnormal user experience. Please use your expertise to score the given analytical text from the perspectives of **professionalism** and **readability of the sentences**, providing reasons for your score.

Scoring criteria: The scores are divided into 5 levels:
1: Contains many confusing contents or grammatical errors.
2: The text has no grammatical errors, and the sentences are smooth, but the content involving the professional field of wireless communication contains technical errors or is hard to understand.
3: The text is clear, detailed, uses accurate professional terminology in wireless communication, is logically coherent, and contains no obvious reasoning errors.\n\n'''
        self.role += self.output_format


    def getPrompt(self, text):
        output = f'Text for analysis: {text}'
        return output
    
    def getRate(self, text):
        # output: the output of LLM
        prefix = prefix_nl_explanation
        suffix = suffix_nl_explanation
        text = text[text.find(prefix)+len(prefix):text.find(suffix)]
        text = text.strip()

        prompt = self.getPrompt(text)
        result = get_response(self.role, prompt)

        prefix = 'Score:'
        suffix = 'Reasons:'
        result = result[result.find(prefix)+len(prefix):result.find(suffix)]
        result = result.strip()
        score = int(result)
        return score, text
        


class LCSRate:
    def __init__(self) -> None:
        self.__constructGraph()

    def getRate(self, output):
        # output: the output of LLM
        output_chain = self.getChain(output)
        start_node = self.kpi_list.index(output_chain[0])
        end_node = self.kpi_list.index(output_chain[-1])
        shorest_paths = list(nx.all_shortest_paths(self.G, start_node, end_node))
        output_path = [self.kpi_list.index(i) for i in output_chain]
        # 评分1: LCS，越高越好
        max_ = -1e10
        rate = 0
        for path in shorest_paths:
            tmp = self.__computeLCS(path, output_path)
            if len(tmp) > max_:
                max_ = len(tmp)
                rate = max_ / len(path) * 5 # rate: [0,5]
        # 评分1: LCS，越高越好，以占真实路径的比例为最终得分

        return rate, output_chain


    def getChain(self, text):
        prefix = prefix_chain
        suffix = suffix_chain
        # get the text between prefix and suffix
        text = text[text.find(prefix)+len(prefix):]
        text = text.strip()
        lines = text.split('\n')
        chain = []
        for line in lines:
            # get the last word
            word = line.split(' ')[-1]
            if (word.startswith('[') and word.endswith(']')) or (word.startswith('(') and word.endswith(')')):
                word = word[1:-1]
            # if word not in self.kpi_list, find the most same item to replace it
            if word not in self.kpi_list:
                word = self.__find_most_similar(word, self.kpi_list)
            chain.append(word)
        return chain
    
    def __constructGraph(self):
        self.kpi_list = ['DLUserThrpAvgwithoutLastTTI(Mbps)', 'DlSpecEff', 'Rank2Ratio', 'DLMCSAvg', 'DLIBLER', 'DLRBLER', 'ucDlWbCqiCode1', 'ucDlWbCqiCode2', 'ServiceCellRSRP', 'ulTaValue']
        self.G = nx.DiGraph()
        edges_name = [('DlSpecEff', 'DLUserThrpAvgwithoutLastTTI(Mbps)'), 
                      ('Rank2Ratio', 'DlSpecEff'), 
                      ('DLMCSAvg', 'DlSpecEff'),
                      ('DLIBLER', 'DlSpecEff'), 
                      ('DLRBLER', 'DlSpecEff'), 
                      ('ucDlWbCqiCode1', 'Rank2Ratio'),
                      ('ucDlWbCqiCode1', 'DLMCSAvg'),
                      ('ucDlWbCqiCode1', 'DLIBLER'),
                      ('ucDlWbCqiCode1', 'DLRBLER'),
                      ('ucDlWbCqiCode2', 'Rank2Ratio'),
                      ('ucDlWbCqiCode2', 'DLMCSAvg'),
                      ('ucDlWbCqiCode2', 'DLIBLER'),
                      ('ucDlWbCqiCode2', 'DLRBLER'),
                      ('ServiceCellRSRP', 'ucDlWbCqiCode1'),
                      ('ServiceCellRSRP', 'ucDlWbCqiCode2'),
                      ('ulTaValue', 'ServiceCellRSRP')]
        edges = []
        for e in edges_name:
            edges.append((self.kpi_list.index(e[0]), self.kpi_list.index(e[1])))
        self.G.add_edges_from(edges)

    def __computeLCS(self, s1, s2):
        # 构建一个填充了零的矩阵来记录子序列的长度
        matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

        # 使用动态规划填充矩阵
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1] + 1
                else:
                    matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])

        # 从矩阵中恢复LCS
        lcs = []
        i, j = len(s1), len(s2)
        while i > 0 and j > 0:
            if matrix[i][j] == matrix[i - 1][j]:
                i -= 1
            elif matrix[i][j] == matrix[i][j - 1]:
                j -= 1
            else:
                assert s1[i - 1] == s2[j - 1]
                lcs.append(s1[i - 1])
                i -= 1
                j -= 1
        return lcs[::-1]  # 反转以获得正确的顺序
    
    def __find_most_similar(self, target, string_list):
        # 检查列表是否为空
        if not string_list:
            raise ValueError("The string list is empty.")
        # 计算每个字符串与目标字符串的Levenshtein距离
        distances = [Levenshtein.distance(target, s) for s in string_list]
        # 找到最小距离的索引
        min_index = distances.index(min(distances))
        # 返回最相似的字符串
        return string_list[min_index]
    
# test
scorer_lcs = LCSRate()
scorer2, output_chain = scorer_lcs.getRate('''3. List the anomaly propagation steps based on the above analysis:
  - The anomaly first occurred at: ulTaValue
  - Then propagated to: ServiceCellRSRP
  - Then propagated to: Rank2Ratio
  - Then propagated to: DLUserThrpAvgwithoutLastTTI(Mbps) 
  - Eventually leading the anomaly in: DLUserThrpAvgwithoutLastTTI(Mbps)''')


if __name__ == '__main__':
    file = 'RCA/cause.pkl'
    data = pickle.load(open(file, 'rb'))
    final_cause = data['final_cause']
    pred_node = data['pred_node']

    analyzer = RootCauseAnalysisPropmt()
    scorer_grammer = GrammarPrompt()
    scorer_lcs = LCSRate()


    for pred, causes in zip(pred_node, final_cause):
        print(pred, causes, '\n')
        result_text = analyzer.getResult(pred, causes)
        # scorer
        scorer1, explaination = scorer_grammer.getRate(result_text)
        scorer2, output_chain = scorer_lcs.getRate(result_text)

        print('-'*50)
        print(f'LLM: \n{result_text}\n\n')
        print(f'readability: {scorer1}, LCS: {scorer2}\n\n')
        print(f'explanation: {explaination}\n\n')
        print(f'chain: {output_chain}')

        


