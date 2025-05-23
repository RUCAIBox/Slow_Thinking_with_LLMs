import time, json, scrapy, inspect, sys, importlib, re, trafilatura, os, datetime, logging, urllib
import pdfplumber
import requests
from docx import Document
import pandas as pd
from io import BytesIO
from json_repair import repair_json
from contextlib import contextmanager
from htmldate import find_date
from functools import partial
from typing import Any, Dict, Union
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
import ast

logging.basicConfig(level=logging.INFO)
logging.getLogger("watchdog").setLevel(logging.INFO)
logging.getLogger("readability.readability").setLevel(logging.WARNING) 
logging.getLogger("charset_normalizer").setLevel(logging.WARNING)
logging.getLogger("htmldate.extractors").setLevel(logging.WARNING)
logging.getLogger("htmldate.core").setLevel(logging.WARNING)
logging.getLogger("scrapy.middleware").setLevel(logging.WARNING)
logging.getLogger("scrapy.utils.log").setLevel(logging.WARNING)
logging.getLogger("trafilatura.main_extractor").setLevel(logging.WARNING)
logging.getLogger("trafilatura.htmlprocessing").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.WARNING)


def cal_timediff(start):
    now = time.time()
    diff = now - start
    print('-----time consumed: ', diff, '-----')
    return diff

def get_tool_name(tool_call):
    return tool_call.function.name

def get_tool_arg(tool_call):
    return tool_call.function.arguments

def parse_resp_to_json(resp_str):
    if not isinstance(resp_str, str):
        return resp_str

    if not resp_str:
        return {}
    completed_json = repair_json(resp_str)
    if not completed_json:
        return resp_str
    try:
        json_for_fc = json.loads(completed_json)
        if json_for_fc == '' or json_for_fc ==[]:
            return json_for_fc
        elif isinstance(json_for_fc, list):
            return json_for_fc[0]
        else:
            return json_for_fc
    except json.JSONDecodeError as e:
        raise ValueError(f"解析 JSON 时出错: {e}")

def extract_int(index):
    if isinstance(index, int):  
        return index
    elif isinstance(index, str):
        match = re.search(r'\d+', index)
        return int(match.group()) if match else None
    else:
        return None

def load_multiple_dict(resp_str):
    # case: Function(arguments='{"plans": ["2012年获得奥斯卡最佳男主角的演员是英国人吗？"]}{"plans": ["2013年获得奥斯卡最佳男主角的演员是英国人吗？"]}', name='SolvePlan')
    # assume that these dicts share the same key, then merge the values in a list
    merged_dict = {}
    completed_json = repair_json(resp_str)
    pattern = r'\{.*?\}'
    json_strings = re.findall(pattern, completed_json, re.S)
    for json_str in json_strings:
        try:
            json_obj = parse_resp_to_json(json_str)
            for k in json_obj:
                if k in merged_dict:
                    merged_dict[k].append(json_obj[k])
                else:
                    merged_dict[k] = [json_obj[k]] if not isinstance(json_obj[k], list) else json_obj[k]
        except json.JSONDecodeError as e:
            print(f"ERROR IN LOAD_MULTIPLE_DICT: {e}")
            raise
    return merged_dict

def check_ans_valid(resp_str):
    final_resp_json = parse_resp_to_json(resp_str)
    if final_resp_json.get('concise_answer', '') or final_resp_json.get('detailed_answer', ''):
        return True
    return False

def dict_value_isnone(dict):
    isnone = False
    for v in dict.values():
        if not v:
            isnone=True
            break
    return isnone
    
def parse_resp_content(resp):
    if not isinstance(resp, str): # tool call
        return parse_resp_to_json(resp.tool_calls[0].function.arguments)
    else:
        return parse_resp_to_json(resp)
    
def is_complete_json(json_str):
    try:
        json.loads(json_str)
        return True  
    except json.JSONDecodeError:
        return False  
    
def finish_condition(resp):
    if resp and "actions" in resp and resp['actions'] == 'final_response':
        return True
    else:
        return False

@contextmanager
def timeit(description: str):
    start_time = time.time()  
    try:
        yield 
    finally:
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        print(f"{datetime.datetime.now().strftime('%m-%d %H:%M:%S')} - {description} : {elapsed_time:.6f} s")
        print(f"{datetime.datetime.now().strftime('%m-%d %H:%M:%S')} - {description} : {elapsed_time:.6f} s")
        
def load_class_from_string(class_path: str, path=None):
    path_in_sys = False
    if path:
        if path not in sys.path:
            path_in_sys = True
            sys.path.insert(0, path)

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
    finally:
        if path and path_in_sys:
            sys.path.remove(path)


def create_object(config: Union[Dict, Any] = None):
    """Create an instance based on the configuration where 'type' is a
    preserved key to indicate the class (path). When accepting non-dictionary
    input, the function degenerates to an identity.
    """
    if config is None or not isinstance(config, dict):
        return config
    assert isinstance(config, dict) and 'type' in config

    config = config.copy()
    obj_type = config.pop('type')
    if isinstance(obj_type, str):
        obj_type = load_class_from_string(obj_type)
    if inspect.isclass(obj_type):
        obj = obj_type(**config)
    else:
        assert callable(obj_type)
        obj = partial(obj_type, **config)
    return obj

# def remove_think_tags(text: str) -> str:
#     print("=="*70)
#     print("IN REMOVE_THINK_TAG!!!")
#     THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
#     STRAY_CLOSE_TAG = re.compile(r'.*?</think>', re.DOTALL)

#     # Step 1: Remove well-formed <think>...</think>
#     if text:
#         text = re.sub(THINK_TAGS, '', text)
#         # Step 2: If there's an unmatched closing tag </think>,
#         #         remove everything up to and including that.
#         text = re.sub(STRAY_CLOSE_TAG, '', text)
#         return text.strip()
#     else:
#         return text

def remove_think_tags(text: str) -> str:
    if not text:
        return text
    THINK_TAGS = re.compile(r'<think>[^<]*</think>', re.DOTALL)
    STRAY_CLOSE_TAG = re.compile(r'</think>', re.DOTALL)
    
    if re.search(STRAY_CLOSE_TAG, text):
        text = text.split('</think>', 1)[-1]
    
    # 然后处理完整标签
    text = re.sub(THINK_TAGS, '', text)
    
    return text.strip()

def parse_keys(score_dict:dict)->dict:
    original_keys = list(score_dict.keys())  # 假设这是你的字符串
    new_keys = []
    for ori_key in original_keys:
        if isinstance(ori_key, int):
            new_keys.append(ori_key)
        else:
            match = re.search(r'\d+', ori_key)  # 查找连续的数字
            if match:
                number = match.group()  # 提取匹配到的数字
                new_keys.append(int(number))
            else:
                raise ValueError(f"chunk 划分错误")
    parsed_score_dict = {new_key: score_dict[ori_key] for new_key, ori_key in zip(new_keys, original_keys)}
    return parsed_score_dict

# def do_filter_search(tgt, src):
#     count = 0
#     filter_results = {}
#     if src:
#         for key, item in tgt.items():
#             flag = True
#             for src_key, src_item in src.items():
#                 if item['url'] == src_item['url']:
#                     flag = False
#                     break
#             if flag:
#                 filter_results[count] = item
#                 count +=1

#         return filter_results
#     else:
#         return tgt