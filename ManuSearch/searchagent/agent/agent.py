from datetime import datetime
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
from ..tools.tool_collection import ToolCollection
from ..tools.final_answer import FinalAnswerTool
from ..tools.visitpage import VisitPage
from ..tools.websearch import GoogleSearch

my_cache_dir = os.getenv('CACHE_DIR')

class AgentInterface:
    def __init__(
        self,
        google_subscription_key,
        google_search_topk,
        proxy,
        planner_model_name,
        planner_api_base,
        planner_api_key,
        searcher_model_name,
        searcher_api_base,
        searcher_api_key,
        reader_model_name, 
        reader_api_base,
        reader_api_key,
        my_cache_dir,
        temperature,
        top_p, 
        min_p, 
        top_k,
        repetition_penalty, 
        max_new_tokens,
        searcher_same_parameters,
        reader_same_parameters
    ):

        self.date = datetime.now().strftime("The current date is %Y-%m-%d.")
        self.proxy = proxy
        self.search_api_key = google_subscription_key
        self.webpage_cache = WebPageCache(
            cache_dir=my_cache_dir,
        )
        planner_gen_params = {
            "temperature": temperature,
            "top_p": top_p, 
            "min_p": min_p, 
            "top_k": top_k,
            "repetition_penalty": repetition_penalty, 
            "max_new_tokens": max_new_tokens,
        }

        self.planner_model = GPTAPI(
            model_type=planner_model_name,
            key=planner_api_key,
            api_base=planner_api_base,
            **planner_gen_params
        )
        
        if searcher_same_parameters:
            searcher_gen_params = planner_gen_params
        else:
            searcher_gen_params  = {
            "temperature": 0.6,
            "top_p": 0.95, 
            "min_p": 0.0, 
            "top_k": 30,
            "repetition_penalty": 1.0, 
            "max_new_tokens": 8192,
        }

        self.searcher_model = GPTAPI(
            model_type=searcher_model_name,
            key=searcher_api_key,
            api_base=searcher_api_base,
            **searcher_gen_params
        )

        if reader_same_parameters:
            reader_gen_params = planner_gen_params
        else:
            reader_gen_params  = {
            "temperature": 0.6,
            "top_p": 0.95, 
            "min_p": 0.0, 
            "top_k": 30,
            "repetition_penalty": 1.0, 
            "max_new_tokens": 8192,
        }

        self.reader_model = GPTAPI(
            model_type= reader_model_name,
            key=reader_api_key,
            api_base=reader_api_base,
            **reader_gen_params
        )

        self.searcher_tools = ToolCollection(
            GoogleSearch(api_key=google_subscription_key, topk=google_search_topk), 
            FinalAnswerTool()
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

        self.reader = Reader(
            llm=self.reader_model,
            template=self.date,
            summary_prompt = READER_SUMM_PROMPT_CN,
            extract_prompt = READER_EXTRACT_PROMPT_CN,
            webpage_cache=self.webpage_cache,
            proxy=self.proxy,
            search_api_key=self.search_api_key[0]
        )
        self.searcher = Searcher(
            user_context_template=searcher_context_template_cn,
            user_input_template=searcher_input_template_cn,
            template=self.date,
            system_prompt=SEARCHER_PROMPT_CN,
            llm=self.searcher_model,
            reader=self.reader,
            collected_tools=self.searcher_tools,
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
            iterative_prompt=PLANNER_ITERATIVE_PROMPT_CN.format(current_date = datetime.now().strftime("%Y-%m-%d"))
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