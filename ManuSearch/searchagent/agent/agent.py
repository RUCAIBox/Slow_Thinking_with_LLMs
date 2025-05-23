from datetime import datetime
from dotenv import load_dotenv
import os, traceback, sys
from ..models.planner import Planner
from ..models.searcher import Searcher
from ..models.reader import Reader
from ..models.recorder import Recorder
from ..models.basellm import GPTAPI
from ..models.searchagent import SearchAgent
from ..prompt.planner import *
from ..prompt.reader import *
from ..prompt.searcher import *
from ..prompt.agent import *
from ..utils.cache import WebPageCache

planner_model_type =  os.getenv('PLANNER_MODEL_TYPE')
searcher_model_type =  os.getenv('SEARCHER_MODEL_TYPE')
reader_model_type = os.getenv("READER_MODEL_TYPE") 

searcher_api_base =  os.getenv('SEARCHER_API_BASE')
planner_api_base =  os.getenv('PLANNER_API_BASE')
reader_api_base =  os.getenv('READER_API_BASE')

searcher_api_key =  os.getenv('SEARCHER_API_KEY')
planner_api_key =  os.getenv('PLANNER_API_KEY')
reader_api_key =  os.getenv('READER_API_KEY')

cache_dir = os.getenv('CACHE_DIR')

class AgentInterface:
    def __init__(self):
        self.date = datetime.now().strftime("The current date is %Y-%m-%d.")

        self.webpage_cache = WebPageCache(
            cache_dir="/opt/aps/workdir/SearchAgent/WebRAG/cache",
        )

        self.main_model = GPTAPI(
            model_type=main_model_type,
            key=env_gpt_api_keys_list,
            api_base=main_model_api_base,
            max_new_tokens=8192,
            temperature=0.6,
        )

        self.planner_model = GPTAPI(
            model_type=planner_model_type,
            key=planner_api_key,
            api_base=planner_api_base,
            max_new_tokens=8192,
            temperature=0.6,
        )
        self.searcher_model = GPTAPI(
            model_type=searcher_model_type,
            key=searcher_api_key,
            api_base=searcher_api_base,
            max_new_tokens=8192,
            temperature=0.6
        )
        self.reader_model = GPTAPI(
            model_type= reader_model_type,
            key=reader_api_key,
            api_base=reader_api_base,
            max_new_tokens=8192,
            temperature=0.6,
        )
        self.deep_reasoning_model = GPTAPI(
            model_type= deep_reasoning_model_type,
            key=env_ds_api_keys_list,
            api_base=deep_reasoning_model_api_base,
            max_new_tokens=8192,
            temperature=0.6,
        )


    def get_answer(self, message: str, solve_method='iterative', deep_reasoning=False, history=''):
        
        def get_ascii_part(input_text):
            english_count = 0
            total_count = len(input_text)
            for char in input_text:
                if char.isascii() and char.isalpha():
                    english_count += 1
            return english_count/total_count
            
        use_en = get_ascii_part(message) > 0.5
        if deep_reasoning:
            self.planner_model = self.deep_reasoning_model
            self.searcher_model = self.deep_reasoning_model
        else:
            pass

        self.reader = Reader(
            llm=self.reader_model,
            template=self.date,
            summary_prompt = READER_SUMM_PROMPT_CN,
            extract_prompt = READER_EXTRACT_PROMPT_CN,
            webpage_cache=self.webpage_cache
        )
        self.searcher = Searcher(
            user_context_template=searcher_context_template_cn,
            user_input_template=searcher_input_template_cn,
            template=self.date,
            system_prompt=SEARCHER_PROMPT_CN,
            llm=self.searcher_model,
            reader=self.reader,
        )
        self.recorder = Recorder(
            action=None
        )
        self.planner= Planner(
            llm=self.planner_model,
            template=self.date,
            system_prompt=PLANNER_ITERATIVE_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d")),
        )

        self.agent = SearchAgent(
            planner=self.planner,
            searcher=self.searcher,
            recorder=self.recorder,
            max_turn=10,
            llm=self.planner_model,
            iterative_prompt=PLANNER_ITERATIVE_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d")),
            sequential_prompt=PLANNER_SEQUENTIAL_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d"))
        )

        if use_en:
            self.planner.system_prompt = PLANNER_ITERATIVE_PROMPT_EN
            self.planner.agent.system_prompt = PLANNER_ITERATIVE_PROMPT_EN
            self.reader.summary_prompt = READER_SUMM_PROMPT_EN
            self.reader.extract_prompt = READER_EXTRACT_PROMPT_EN
            self.searcher.user_context_template = searcher_context_template_en
            self.searcher.user_input_template = searcher_input_template_en
            self.searcher.system_prompt = SEARCHER_PROMPT_EN
            self.searcher.agent.system_prompt = SEARCHER_PROMPT_EN
            self.context_prompt = CONTEXT_PROMPT_EN
            self.agent.iterative_prompt = PLANNER_ITERATIVE_PROMPT_EN
        else:
            self.planner.system_prompt = PLANNER_ITERATIVE_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d"))
            self.reader.summary_prompt = READER_SUMM_PROMPT_CN
            self.reader.extract_prompt = READER_EXTRACT_PROMPT_CN
            self.searcher.user_context_template = searcher_context_template_cn
            self.searcher.user_input_template = searcher_input_template_cn
            self.searcher.system_prompt = SEARCHER_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d"))
            self.context_prompt = CONTEXT_PROMPT_CN
            self.agent.iterative_prompt = PLANNER_ITERATIVE_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d"))

        if history:
            context = self.context_prompt.format(history_qa = history, question = message)
        else:
            context = message
        print('*****'*5, solve_method, deep_reasoning, '*****'*5)
        
        try:
            for step in self.agent.forward(context, mode=solve_method):
                yield step, use_en
                
        except Exception as e:
            print('=='*40)
            print('agent error: ', e)
            print('Stack trace:', traceback.format_exc()) 
            print('=='*40)
            raise