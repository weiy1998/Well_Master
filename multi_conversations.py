##
# 井控微调数据集的预处理-转为json格式
# 1.读取数据
# 2.数据清洗
# 3.格式转换
# author: WQY
# ##
# 选择题多轮问答
import json
import re

import pandas as pd


def exercise_to_json(input_file_path):
    # 读取Excel文件
    df = pd.read_excel(input_file_path, sheet_name=None)

    conversations = []
    # 遍历DataFrame的每一行
    for index, row in df['选择题'].iterrows():
        # 创建一个空的列表来存储问答对
        qa_pairs = []
        # 获取问题和正确答案
        question = row['题目']
        question = re.sub(r'\(\s*(.*?)\s*\)', r'(\1)', question)
        question = question.replace('(', '（').replace(')', '）')

        correct_answer = row['题目+答案']

        # 获取选项列
        option_columns = [col for col in df['选择题'].columns if '易错选项' in col]
        option_columns.insert(0,'答案')

        # 遍历每个选项
        for option in option_columns:
            if pd.notnull(df['选择题'].loc[index, option]) and option == '答案':
                # 构建问答对
                qa_pair = {
                    'question': question.replace('（）', f'{row[option]}').replace('。', '?'),
                    'answer': f'是的！{correct_answer}'
                }
            elif pd.notnull(df['选择题'].loc[index, option]) and option != '答案':
                qa_pair = {
                    'question': question.replace('（）', f'{row[option]}').replace('。', '?'),
                    'answer': f'不对！{correct_answer}'
                }
            else:
                continue
            # 将问答对添加到列表中
            qa_pairs.append(qa_pair)
        conversation = {
            "conversation": [
                {
                    "system": "现在你是一位专业的井控领域的专家，你的名字叫井控专家，你的说话方式是严谨、熟练的，并且总是解答专业的井控知识的问题。",
                    "input": qa["question"],
                    "output": qa["answer"]
                } for qa in qa_pairs
            ]
        }

        conversations.append(conversation)
    # 将列表转换为JSON格式
    # 移除除了第一个元素之外的 "system" 字段
    for conversation in conversations:
        for item in conversation["conversation"][1:]:
            item.pop("system", None)
    json_data = json.dumps(conversations, indent=4, ensure_ascii=False)

    # 将JSON数据保存到文件
    with open('./data/多轮问答选择题.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


if __name__ == '__main__':
    # 指定输入输出文件路径
    input_file_path1 = r"D:\project\井控大模型\微调数据\微调数据\习题集（分类后）.xlsx"
    exercise_to_json(input_file_path1)
