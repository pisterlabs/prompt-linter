"""
Language Model Calling, Generating Rubrics
"""

# Author: Wosida, Rvosuke
# Date: 2023/10/5

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# 设置api密钥
key = "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"


# 调用api
def claude(calculated_iou: float = 0,
           calculated_image_similarity: float = 0,
           calculated_keypoint_matching: float = 0,
           api_key=key):
    text = f"""
    Based on the analysis of the calligraphic copy, the following aesthetic features have been extracted:

1. Intersection over Union (IoU): {calculated_iou}
   - IoU measures the fullness of the characters and the fidelity of the copy to the template. 

2. Image Similarity: {calculated_image_similarity}
   - This metric evaluates the visual similarity between the copy and the template, indicating how well the copy captures the essence of the template.

3. Keypoint Matching: {calculated_keypoint_matching}
   - This assesses the precision of the brushstrokes, providing insight into the skill level and attention to detail of the artist.

Could you please generate a comprehensive review and guidance based on these features by Chinese? 
The review should include specific comments on each feature and overall advice on how to improve the aesthetic quality of the calligraphy.
The above three indicators range from 1 to 10. If 0 appears, the indicator is ignored.
But please do not generate any sentences that are not related to the comments, and there is no need for reasoning.
You should give the comments directly in one or two sentences like a teacher.

Your answer should look like the following example:
"字体笔画过于单薄,应当注重运笔的力度,整体布局和结构基本遵循范本,但还需提高对细节的把握,笔画结束处缺乏收.请加油!"
Please give your comments directly and do not include the following content in your answer
" 您好,根据您提供的特征分析,我给出以下评价和建议:"
"""
    anthropic = Anthropic(api_key=api_key)
    completion = anthropic.completions.create(
        model="claude-2",  # 选择模型
        max_tokens_to_sample=1000,  # 设置最大生成长度
        prompt=f"{HUMAN_PROMPT}{text}{AI_PROMPT}",  # 设置prompt
    )
    answer = completion.completion
    # 将claude回话添加到列表，实现上下文解析
    # text_input.append({"role": "claude", "content": answer})
    # 打印输出
    # print(answer)

    return answer


if __name__ == '__main__':
    # 创建对话空列表
    conversation = []
    while True:
        prompt = input("user（输入q退出）:")
        if prompt == "q":
            break
        # 将用户输入添加到列表
        conversation.append({"role": "user", "content": prompt})
        # 调用claude函数
        claude(5.0, 7.0, 1.5)
