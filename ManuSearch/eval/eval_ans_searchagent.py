import os,sys
p1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(p1)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from searchagent.utils.utils import remove_think_tags
import json
import re
from openai import OpenAI

# 环境变量配置
os.environ["OPENAI_API_KEY"] = "sk-6QzD4g7HAF6H9cpCKIS2pWe4e6OYaEAo1RQevozs3rk57SkE"
os.environ["OPENAI_API_BASE"] = "https://us.ifopen.ai/v1/"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

# 用于生成 GPT 输出的函数
def generate(messages, model_name):
    response = client.chat.completions.create(
        **{
            "model": model_name,
            "messages": messages,
            "max_tokens": 2048,
        }
    )
    response = response.choices[0].message.content
    return response

# 数据验证和结果保存
def validate_data(file):
#     PROMPT = """You will receive a question along with a reference answer and a predicted answer. Your task is to evaluate the accuracy of the predicted answer and provide a concise explanation.

# Compare the predicted answer to the reference answer to determine its correctness.

# **Guidelines**
# - The criteria for evaluating the predicted answer should not be overly strict. If the predicted answer's meaning aligns closely with that of the reference answer, it can be deemed correct.
# - For each question, provide a brief explanation of your reasoning, followed by "Correct" or "Incorrect." Include your final assessment within <assessment> tags.

# **Output Format**
# [Explanation]: Provide a brief explanation supporting your judgment.
# [Assessment]: Provide your assessment **within <assessment> tags**.

# Here is the question:
# {question}

# Here is the reference answer:
# {reference}

# Here is the predicted answer:
# {prediction}
# """
    PROMPT = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
    '''

    # 获取所有以_finished结尾的jsonl文件
    print("Begin: ",file)
    file_path =file
    print("Now file:",file_path)

    # 初始化计数器
    valid_num = 0
    correct_num = 0
    incorrect_num = 0
    result_data = []  # 用来存储每个对象的处理结果

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue

            valid_num += 1
            prediction = obj.get("answer", "I don't know")
            if isinstance(prediction, dict):
                prediction = prediction.get("content")
                if isinstance(prediction, dict):
                    prediction=prediction.get("concise_answer","")
            else:
                prediction = remove_think_tags(prediction)
            question = obj.get("question", "")
            answer = obj.get("gold", [])
            print("=="*70)
            print("Question:",question)
            print(prediction)
            print(answer)

            gpt4o_input = PROMPT.format(question = question,reference=answer, prediction=prediction)
            messages = [{'role': 'user', 'content': gpt4o_input}]

            # 获取GPT模型的评判结果
            model_output = generate(messages, 'gpt-4o')

            if "false" in model_output.lower():
                is_correct = False
            else:
                is_correct = True

            if is_correct:
                correct_num += 1
            else:
                incorrect_num += 1

            obj['check_ans'] = model_output
            print("no.",valid_num,": ",model_output)

            # 将带有评判结果的对象加入到结果列表中
            result_data.append(obj)

    # 计算准确率
    if valid_num > 0:
        accuracy = correct_num / valid_num * 100
    else:
        accuracy = 0

    # 输出结果
    print(f"File: {file}")
    print(f"Valid objects: {valid_num}")
    print(f"Correct objects: {correct_num}")
    print(f"Incorrect objects: {incorrect_num}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    # 保存到新的JSONL文件，名称与原文件相同
    output_file_path = file_path.replace('.jsonl', '_with_check_ans.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for obj in result_data:
            output_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    # break

# 使用示例
files = [
"/opt/aps/workdir/SearchAgent_Copy/WebRAG/searchagent/test/SearchAgent_self_test_CN/qwen-32B.jsonl"
]
for file in files:
    validate_data(file)

