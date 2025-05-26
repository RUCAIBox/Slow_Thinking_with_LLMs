from ..utils.utils import *
from ..utils.cache import WebPageCache
from ..models.basellm import GPTAPI, BaseStreamingAgent
from ..tools.visitpage import VisitPage
from concurrent.futures import ThreadPoolExecutor, as_completed 
import os, ast
import concurrent.futures

class Reader(BaseStreamingAgent):
    def __init__(self, llm:GPTAPI, webpage_cache, summary_prompt, extract_prompt, search_api_key, proxy, **baseconfig):
        self.llm = llm
        self.summary_prompt = summary_prompt
        self.extract_prompt = extract_prompt
        self.input_prompt = """##Publish Date:{date}
        ##Web title:{title}
        ##Web content:{content}"""
        self.visitpage = VisitPage(api_key=search_api_key, timeout=1, proxy=proxy)
        self.webpage_cache = webpage_cache

        super().__init__(llm, **baseconfig)


    def get_llm_summ(self, search_results:dict, question, user_query, search_intent, current_query):
        url2id = {value['url']: key for key,value in search_results.items()}
        select_urls = []
        for key in url2id.keys():
            if key:
                select_urls.append(key)

        # First read the stored url from cache
        cached_results = {} 
        if self.webpage_cache:
            for key in select_urls:
                success, content = self.webpage_cache.get_content(url=key)
                if success:
                    cached_results[url2id[key]] = content 
                    select_urls.remove(key)

        cleaned_tool_return = {}

        # If there is an unstored url, access it
        if select_urls:
            with timeit("reader fetch all pages"):
                tool_return = self.visitpage.execute(select_urls=select_urls, search_results=search_results, url_to_chunk_score = None, webpage_cache=self.webpage_cache)# 筛选出提取出来正文的进行进一步summary

            if not tool_return:
                print("Visitpage couldn't execute")
                return search_results, "Visitpage couldn't execute"

            cleaned_tool_return = self.extract_text(tool_return=tool_return)
            # cache accessed urls        
            for items in cleaned_tool_return.values():
                self.webpage_cache.store_content(url=items['url'], data=items)

        # merge cache hit and miss urls
        if cleaned_tool_return:
            cleaned_tool_return.update(cached_results)
        else:
            cleaned_tool_return = cached_results
        
        messages, url_to_chunks={}, {}
        system_prompt = self.summary_prompt.format(current_plan=question, user_query = user_query, search_intent=search_intent, current_query=current_query)
        for item in cleaned_tool_return.values():
            url = item['url']
            url_to_chunks[url] = item['content']
            if 'content' not in item or not item['content']:
                continue
            chunked_str = '=========='.join([f"Chunk {key}:{value}" for key, value in item['content'].items()])
            chunked_str = chunked_str[:16192]
            if 'title' not in item:
                item['title'] = ""
            content = self.input_prompt.format(date=item['date'], title=item['title'], content=chunked_str)
            chatbox=[
                {"role": 'system', 'content': system_prompt},
                {'role': 'user', 'content': content}
            ]
            messages[url]=chatbox


        with timeit("reader llm summ"):
            url2summ = {}
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {
                    executor.submit(self.llm.chat, chatbox): url
                    for url, chatbox in messages.items()
                }
                try:
                    for future in concurrent.futures.as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            # Set the timeout period
                            ret = future.result(timeout=10)
                            llm_summ = ret.content
                            url2summ[url] = llm_summ

                        except concurrent.futures.TimeoutError:
                            print(f"Task for {url} timed out")
                            url2summ[url] = "Timeout Error"
                        
                        except Exception as e:
                            print(f"Task for {url} generated an exception: {e}")
                            # Handle other possible exceptions
                            url2summ[url] = f"Error: {str(e)}"
                except concurrent.futures.TimeoutError:
                    raise ValueError("concurrent.futures TimeoutError!")

        llm_summs = url2summ # {url: summ}
        for key in llm_summs:
            reader_json = parse_resp_to_json(llm_summs[key])
            try:
                llm_summs[key] = reader_json.get('related_information', '')
            except:
                pass
        for page in search_results.values():
            if page['url'] in llm_summs:
                page['content'] = llm_summs[page['url']]
            else:
                page['content'] = ""
        return search_results, None # {url: {chunk_dict, scores}}

    def extract_text(self, tool_return):
        messages = {}
        for item in tool_return.values():
            url = item['url']
            messages[url] = []
            if 'content' not in item or not item['content']:
                continue
            if isinstance(item['content'], str):
                chunked_str = item['content']
            chunked_str = ''.join(list(item['content'].values()))
            if len(chunked_str) > 128000:
                chunked_str = chunked_str[:64000]
            if len(chunked_str) > 16192: 
                content = chunked_str[:16192] 
                i=0
                while 16192*(i+1) <= len(chunked_str):
                    content = chunked_str[16192*i:16192*(i+1)] 
                    i += 1
                    chatbox=[
                        {"role": 'system', 'content': self.extract_prompt},
                        {'role': 'user', 'content': content}
                    ]
                    messages[url].append(chatbox)
            else:
                content = chunked_str
                chatbox=[
                    {"role": 'system', 'content': self.extract_prompt},
                    {'role': 'user', 'content': content}
                ]
                messages[url].append(chatbox)

        webtexts = {}
        inputs = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            for url, chatboxes in messages.items():
                for chatbox in chatboxes:
                    inputs.append((url, chatbox))
            future_to_url = {
                executor.submit(self.llm.chat, chatbox): url
                for url, chatbox in inputs
            }
            try:
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                    
                        ret = future.result(timeout=10)
                        text = ret.content
                        if url in webtexts:
                            webtexts[url] += text
                        else:
                            webtexts[url] = text

                    except concurrent.futures.TimeoutError:
                        print(f"Task for {url} timed out")
                        webtexts[url] = "Timeout Error"

                    except Exception as e:
                        print(f"Task for {url} generated an exception: {e}")
                        webtexts[url] = f"Error: {str(e)}"
                        
            except concurrent.futures.TimeoutError:
                raise ValueError("concurrent.futures TimeoutError!")

        for item in tool_return.values():
            url = item['url']
            if 'content' not in item or not item['content']:
                continue
            if url in webtexts and webtexts[url]:
                item['content'] = self.chunk_content(webtexts[url], chunk_size=512)

        return tool_return
    
    def chunk_content(self, text, chunk_size=512):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}
        return chunk_dict