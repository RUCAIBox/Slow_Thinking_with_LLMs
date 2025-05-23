import json



if __name__ == "__main__":

    file_path = "/opt/aps/workdir/SearchAgent_Copy/WebRAG/searchagent/test/rerun/r1_r1_ans_5_6_2300_with_check_ans.jsonl"
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

                if int(obj['level']) == 1:
                    level_1 += 1
                    if "true" in obj['check_ans'].lower():
                        level_1_correct += 1
                    else:
                        pass
                elif int(obj['level']) == 2:
                    level_2 += 1
                    if "true" in obj['check_ans'].lower():
                        level_2_correct += 1
                    else:
                        pass
                elif int(obj['level']) == 3:
                    level_3 += 1
                    if "true" in obj['check_ans'].lower():
                        level_3_correct += 1
                    else:
                        pass
            except json.JSONDecodeError:
                print("Error:",line)
                continue
    print("=="*40)
    print(f"Level 1:{level_1},Level 1 correct:{level_1_correct}")
    print(f"Level 1 Accuracy:{level_1_correct*100/level_1:.2f}%")
    print("=="*40)
    print(f"Level 2:{level_2},Level 2 correct:{level_2_correct}")
    print(f"Level 2 Accuracy:{level_2_correct*100/level_2:.2f}%")
    print("=="*40)
    print(f"Level 3:{level_3},Level 3 correct:{level_3_correct}")
    print(f"Level 3 Accuracy:{level_3_correct*100/level_3:.2f}%")