##
# 井控微调数据集的预处理-转为json格式
# 1.读取数据
# 2.数据清洗
# 3.格式转换
# author: WQY
# ##
# 问答题单轮问答
import json

import pandas as pd


def drilling_well_control_to_json(input_file_path):
    # 读取Excel文件
    df = pd.read_excel(input_file_path)

    # 创建一个空的列表来存储问答对
    conversations = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 获取问题和正确答案
        question = row['问题']

        correct_answer = row['答案1']


        # 构建问答对
        qa_pair = {
            'question': question,
            'answer': f'{correct_answer}'
        }

        conversation = {
            "conversation": [
                {
                    "system": "现在你是一位专业的井控领域的专家，你的名字叫井控专家，你的说话方式是严谨、熟练的，并且总是解答专业的井控知识的问题。",
                    "input": qa_pair["question"],
                    "output": qa_pair["answer"]
                }
            ]
        }
        conversations.append(conversation)
    # 将列表转换为JSON格式
    json_data = json.dumps(conversations, indent=4, ensure_ascii=False)

    file_name = input_file_path.split('\\')[-1].split('.')[0] + '.json'
    print(file_name)
    # 将JSON数据保存到文件
    with open(f'./data/单轮{file_name}', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

if __name__ == "__main__":
    # 指定输入输出文件路径
    input_file_path2 = r"D:\project\井控大模型\微调数据\微调数据\钻井井控技术问答.xlsx"
    input_file_path3 = r"D:\project\井控大模型\微调数据\微调数据\钻井设备问答.xlsx"
    input_file_path4 = r"D:\project\井控大模型\微调数据\微调数据\钻井液工艺技术问答.xlsx"

    input_file_paths = [input_file_path2, input_file_path3, input_file_path4]
    for input_file_path in input_file_paths:
        drilling_well_control_to_json(input_file_path)