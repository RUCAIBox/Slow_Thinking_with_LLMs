import time
import logging
import random
import warnings
import json
import os
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
from http.client import HTTPSConnection
from cachetools import TTLCache, cached
from ..tools.basetool import BaseTool
import concurrent.futures

class GoogleSearch(BaseTool):
    """
    Wrapper around the Serper.dev Google Search API.

    To use, you should pass your serper API key to the constructor.

    Args:
        api_key (List[str]): API KEY to use serper google search API.
            You can create a free API key at https://serper.dev.
        search_type (str): Serper API supports ['search', 'images', 'news',
            'places'] types of search, currently we only support 'search' and 'news'.
        topk (int): The number of search results returned in response from api search results.
        **kwargs: Any other parameters related to the Serper API. Find more details at
            https://serper.dev/playground
    """

    name: str = "GoogleSearch"
    description: str = "Performs a google web search for your query and your search intent then returns a string of the top search results."
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items":{
                    "type": "string"
                },
                "description": "The search queries to perform."
            },
            "intent":{
                "type": "array",
                "items":{
                    "type": "string"
                },
                "description": "The detailed intent of the performing this search."
            },

        },
        "required": ["query","intent"],
        "additionalProperties": False
    }
    api_key: List[str]
    topk: int
    black_list: List[str]
    key_ctr : int
    invalid_keys : set
    def __init__(
        self,
        api_key: List[str] = None,
        topk: int = 5,
        black_list: List[str] = [
            'enoN',
            'youtube.com',
            'researchgate.net',
        ],
        key_ctr: int = 0,
        invalid_keys: set = set()
    ):

        super().__init__(api_key=api_key, topk=topk, black_list=black_list, key_ctr=key_ctr, invalid_keys=invalid_keys)

    def __hash__(self):
        return hash(self.description)

    def __eq__(self, other):
        if isinstance(other, GoogleSearch):
            return self.description == other.description
        return False


    def execute(self, *, intent:str, query: Union[str, List[str]]) -> dict:
        """Google search API
        Args:
            query (List[str]): list of search query strings
        """
        queries = query if isinstance(query, list) else [query]
        _search_results = {}

        with ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(self._search, q): q for q in queries}

            for future in concurrent.futures.as_completed(future_to_query, timeout=60):
                query = future_to_query[future]
                try:
                    results = future.result(timeout=20)
                except Exception as exc:
                    warnings.warn(f'{query} generated an exception: {exc}')
                else:
                    for result in results.values():
                        if result['url'] not in _search_results:
                            _search_results[result['url']] = result
                        else:
                            _search_results[result['url']]['summ'] += f"\n{result['summ']}"
                    # yield search_results
                
        for item in _search_results.values():
            if not item['url']:
                clues = f"This is an official summary of Wikipedia's information as summarized by the authoritative Google: {item['summ']}"
                item['summ'] = clues
        search_results = {idx: result for idx, result in enumerate(_search_results.values())}
        # yield self.search_results
        return search_results


    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def _search(self, query: str, max_retry: int = 3) -> dict:

        max_num_retries, errmsg = 0, ''
        while max_num_retries < max_retry:

            with Lock():
                if len(self.invalid_keys) == len(self.api_key):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.api_key):
                        self.key_ctr = 0

                    if self.api_key[self.key_ctr] not in self.invalid_keys:
                        break
            
            api_key = self.api_key[self.key_ctr]

            try:
                response = self._call_serper_api(api_key, query)
                if 'result' not in response and 'organic' not in response:
                    time.sleep(1)
                    response = self._call_serper_api(api_key=api_key,query=query)
                
                if 'status' in response:
                    if response['statusCode'] == 400:
                        self.invalid_keys.add(api_key)
                        warnings.warn(f'Retry {max_num_retries + 1}/{max_retry} due to error: {e}')
                        continue
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(f'Retry {max_num_retries + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
            
            max_num_retries += 1
        raise Exception('Failed to get search results from Google Serper Search after retries.')


    def _call_serper_api(self, api_key, query: str) -> dict:

        conn = HTTPSConnection("google.serper.dev", timeout=30)
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            return json.loads(data.decode("utf-8"))
        finally:
            conn.close()


    def _filter_results(self, results: List[tuple]) -> dict:
        filtered_results = {}
        count = 0
        for url, snippet, title in results:
            if all(domain not in url for domain in self.black_list) :
                filtered_results[count] = {
                    'url': url,
                    'summ': json.dumps(snippet, ensure_ascii=False),
                    'title': title,
                }
                count += 1
                if count >= self.topk:
                    break
        return filtered_results

    def _parse_response(self, response: dict) -> dict:
        raw_results = []
        if isinstance(response, str):
            import ast
            response = ast.literal_eval(response)
        if response.get('answerBox'):
            answer_box = response.get('answerBox', {})
            if answer_box.get('answer'):
                raw_results.append(('', answer_box.get('answer'), ''))
            elif answer_box.get('snippet'):
                raw_results.append(('', answer_box.get('snippet').replace('\n', ' '), ''))
            elif answer_box.get('snippetHighlighted'):
                raw_results.append(('', answer_box.get('snippetHighlighted'), ''))

        if response.get('knowledgeGraph'):
            kg = response.get('knowledgeGraph', {})
            description = kg.get('description', '')
            attributes = '. '.join(f'{attribute}: {value}' for attribute, value in kg.get('attributes', {}).items())
            raw_results.append(
                (
                    kg.get('descriptionLink', ''),
                    f'{description}. {attributes}' if attributes else description,
                    f"{kg.get('title', '')}: {kg.get('type', '')}.",
                )
            )

        if 'result' in response:
            for result in response['result']:
                description = result.get('body', '')
                attributes = '. '.join(
                    f'{attribute}: {value}' for attribute, value in result.get('attributes', {}).items()
                )
                raw_results.append(
                    (
                        result.get('href', ''),
                        f'{description}. {attributes}' if attributes else description,
                        result.get('title', ''),
                    )
                )
        elif 'organic' in response: # for serper.dev free
            for result in response['organic'][: self.topk]:
                description = result.get('snippet', '')
                raw_results.append(
                    (
                        result.get('link', ''),
                        description,
                        result.get('title', ''),
                    )
                )
        else:
            print(f"warning when doing search:{response}\napi key:{self.api_key}")

        return self._filter_results(raw_results)