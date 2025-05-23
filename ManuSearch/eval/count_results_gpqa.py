import json



if __name__ == "__main__":

    file_path = "/opt/aps/workdir/SearchAgent/WebRAG/searchagent/test/gpqa/qwq_qwq_ans_with_check_ans.jsonl"
    datas = []
    level_1 = 0
    level_2 = 0
    level_3 = 0
    level_1_correct = 0
    level_2_correct = 0
    level_3_correct = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)

                if obj['domain'] == 'Physics':
                    level_1 += 1
                    if "true" in obj['check_ans'].lower():
                        level_1_correct += 1
                    else:
                        pass
                elif obj['domain'] == 'Chemistry':
                    level_2 += 1
                    if "true" in obj['check_ans'].lower():
                        level_2_correct += 1
                    else:
                        pass
                elif obj['domain'] == 'Biology':
                    level_3 += 1
                    if "true" in obj['check_ans'].lower():
                        level_3_correct += 1
                    else:
                        pass
            except json.JSONDecodeError:
                print("Error:",line)
                continue
    print("=="*40)
    print(f"Level 1(Physics):{level_1},Level 1 correct:{level_1_correct}")
    print(f"Level 1(Physics) Accuracy:{level_1_correct*100/level_1:.2f}%")
    print("=="*40)
    print(f"Level 2(Chemistry):{level_2},Level 2 correct:{level_2_correct}")
    print(f"Level 2(Chemistry) Accuracy:{level_2_correct*100/level_2:.2f}%")
    print("=="*40)
    print(f"Level 3(Biology):{level_3},Level 3 correct:{level_3_correct}")
    print(f"Level 3(Biology) Accuracy:{level_3_correct*100/level_3:.2f}%")