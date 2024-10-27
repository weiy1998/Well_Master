##
# 井控微调数据集的预处理
# 1.读取数据
# 2.数据清洗
# 3.格式转换
# author: weiy
# ##
import re

import pandas as pd
import os
import json
import requests
import random
from http import HTTPStatus
import dashscope
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

ZHIPU_API = 'd96b3dc3c1b2ce95e379396f4d5d6da1.B55pm2il4dXs8822'
QWEN_API = 'sk-029ca2483e6749a0aceff839174a9945'
dashscope.api_key = 'sk-029ca2483e6749a0aceff839174a9945'


def read_data(file_path, sheet_name='Sheet1'):
    """_summary_: 读取excel文件

    Args:
        file_path (_type_): 文件路径
        sheet_name (str, optional): 工作表名称. 默认：'Sheet1'

    Returns:
        _type_: dataframe
    """
    data = pd.read_excel(file_path, sheet_name='Sheet1')
    return data


def data_clean(data):
    data = data[['题目', '答案', '易错选项', 'Unnamed: 6', 'Unnamed: 7', '题目+答案']]
    data = data.rename(
        columns={'易错选项': '易错1', 'Unnamed: 6': '易错2', 'Unnamed: 7': '易错3', '题目+答案': '完整描述'})
    data = data.fillna('无')

    return data


def data_format(data, output_path):
    # 将列名与每一行的值组成键值对
    data = data.to_dict(orient='records')

    with open(os.path.join(output_path, 'convert_kv_format.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def call_with_messages(prompt):
    # prompt = input("user:")
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,  # 选择模型
        messages=messages,
        # set the random seed, optional, default to 1234 if not set
        seed=random.randint(1, 10000),
        result_format='message',  # set the result to be "message" format.
    )
    # 之后HTTPStatus为OK时，才是调用成功
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def call_with_prompt(question, answer, wrong1, wrong2, wrong3):
    # prompt = f"Generate a sentence that combines the question '{question}' with the correct answer '{answer}', and also mention the incorrect options: '{wrong1}', '{wrong2}', and '{wrong3}'."
    prompt = f"生成一段话，第一句是将问题 '{question}' 与正确答案 '{answer}' 结合起来组合成一个完整的疑问句；第二句是将 '{question}' 和错误选项 '{wrong1}' 结合起来以错误选项 '{wrong1}' 为答案生成一句完整的疑问句；第三句是将 '{question}' 和错误选项 '{wrong2}' 结合起来以错误选项 '{wrong2}' 为答案生成一句完整的疑问句；第四句首先判断 '{wrong3}' 是否等于 “无”，如果等于“无”，就不生成，如果不等于“无”，则将 '{question}' 和错误选项 '{wrong3}' 结合起来以错误选项 '{wrong3}' 为答案生成一句完整的疑问句"

    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_turbo,  # 选择模型
        prompt=prompt
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.text.strip()
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.


def create_conversation(question, answer, wrong1, wrong2, wrong3, full_description):
    conversation = [
        {
            "system": "现在你是一位专业的井控领域的专家，你的名字叫井控专家，你的说话方式是严谨、熟练的，并且总是解答专业的井控知识的问题。",
            "input": question,
            "output": full_description
        },
        {
            "input": f"{question} {answer}？",
            "output": f"是的，{full_description}"
        }
    ]

    for wrong in [wrong1, wrong2, wrong3]:
        if wrong != "无":
            conversation.append({
                "input": f"{question} {wrong}？",
                "output": f"不对，{full_description}"
            })

    return conversation


def process_item(item):
    try:
        question = item["题目"]
        answer = item["答案"]
        wrong1 = item["易错1"]
        wrong2 = item["易错2"]
        wrong3 = item["易错3"]
        full_description = item["完整描述"]

        # 生成结合了题目的流畅问句
        combined_question = call_with_prompt(question, answer, wrong1, wrong2, wrong3)

        # 创建对话
        conversation = create_conversation(combined_question, answer, wrong1, wrong2, wrong3, full_description)
        return {'conversation': conversation}

    except Exception as e:
        print(f"Failed to process an item: {e}")


def process_data(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 使用多进程池
    num_workers = cpu_count()  # 使用所有可用的cpu核心
    with Pool(num_workers) as pool:
        # 使用tqdm包裹迭代器以显示进度条
        results = list(tqdm(pool.imap(process_item, data), total=len(data), desc="Processing items", unit="item"))

    # 过滤None值
    conversations = [results for result in results if result is not None]

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(conversations, file, ensure_ascii=False, indent=4)



def exercise_to_json(input_file_path):
    # 读取Excel文件
    df = pd.read_excel(input_file_path, sheet_name=None)
    # 创建一个空的列表来存储问答对
    qa_pairs = []
    # 遍历DataFrame的每一行
    for index, row in df['选择题'].iterrows():
        # 获取问题和正确答案
        question = row['题目']
        question = re.sub(r'\(\s*(.*?)\s*\)', r'(\1)', question)
        question = question.replace('(', '（').replace(')', '）')

        correct_answer = row['题目+答案']

        # 获取选项列
        option_columns = [col for col in df['选择题'].columns if '易错选项' in col]
        option_columns.append('答案')

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

    # 将列表转换为JSON格式
    json_data = json.dumps(qa_pairs, indent=4, ensure_ascii=False)

    # 将JSON数据保存到文件
    with open('./data/选择题.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    # 创建一个空的列表来存储问答对
    qa_pairs = []
    for index, row in df['计算题'].iterrows():
        # 获取问题和正确答案
        question = row['题目']
        correct_answer = row['答案']
        # 构建问答对
        qa_pair = {
            'question': question,
            'answer': f'{correct_answer}'
        }

        qa_pairs.append(qa_pair)

    # 将列表转换为JSON格式
    json_data = json.dumps(qa_pairs, indent=4, ensure_ascii=False)

    # 将JSON数据保存到文件
    with open('./data/计算题.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    # 创建一个空的列表来存储问答对
    qa_pairs = []
    for index, row in df['问答题'].iterrows():
        # 获取问题和正确答案
        question = row['题目']
        correct_answer = row['答案']
        # 构建问答对
        qa_pair = {
            'question': question,
            'answer': f'{correct_answer}'
        }

        qa_pairs.append(qa_pair)

    # 将列表转换为JSON格式
    json_data = json.dumps(qa_pairs, indent=4, ensure_ascii=False)

    # 将JSON数据保存到文件
    with open('./data/问答题.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    # 创建一个空的列表来存储问答对
    qa_pairs = []
    for index, row in df['填空题'].iterrows():
        # 获取问题和正确答案
        question = row['题目']
        correct_answer = row['题目+答案']
        # 构建问答对
        qa_pair = {
            'question': question,
            'answer': f'{correct_answer}'
        }

        qa_pairs.append(qa_pair)

    # 将列表转换为JSON格式
    json_data = json.dumps(qa_pairs, indent=4, ensure_ascii=False)

    # 将JSON数据保存到文件
    with open('./data/填空题.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    # 创建一个空的列表来存储问答对
    qa_pairs = []
    for index, row in df['判断题'].iterrows():
        # 获取问题和正确答案
        question = row['题目']
        correct_answer = row['判断']
        # 构建问答对
        qa_pair = {
            'question': question,
            'answer': f'{correct_answer}'
        }

        qa_pairs.append(qa_pair)

    # 将列表转换为JSON格式
    json_data = json.dumps(qa_pairs, indent=4, ensure_ascii=False)

    # 将JSON数据保存到文件
    with open('./data/判断题.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


if __name__ == "__main__":
    # 指定输入输出文件路径
    input_file_path1 = r"D:\project\井控大模型\微调数据\微调数据\习题集（分类后）.xlsx"
    exercise_to_json(input_file_path1)

