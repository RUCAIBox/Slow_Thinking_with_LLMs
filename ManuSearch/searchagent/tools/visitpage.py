from ..tools.basetool import BaseTool
from ..utils.utils import *
from ..utils.cache import WebPageCache
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from multiprocessing import Process, Manager
from htmldate import find_date
import os, re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from fake_useragent import UserAgent

ua = UserAgent()
# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': ua.random,
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
print(headers)
class VisitPage(BaseTool):

    name: str = "VisitPage"
    description: str = "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    parameters: Optional[dict] = {
        "type": "object",
        "properties": {
            "select_urls": {
                "type": "array",
                "items":{
                    "type": "number"
                },
                "description": "The integar index of the web pages needed to visit for deep reading."
            }
        },
        "required": [
            "select_urls"
        ],
        "additionalProperties": False
    }
    api_key : str
    timeout:  int 
    proxy : str

    def __init__(
            self,
            api_key :str = '',
            timeout : int = 5,
            proxy : str = ''
    ):
        
        super().__init__(api_key=api_key, timeout=timeout, proxy=proxy)


    def execute(self, *, select_urls: List[str], search_results:Dict, url_to_chunk_score: Dict, webpage_cache:WebPageCache) -> dict:
        """get the detailed content on the selected pages.

        Args:
            select_urls (List[str]): list of urls to select. Max number of index to be selected is no more than 4.
        """
                       
        url2id = {value['url']: key for key,value in search_results.items()}
        if not select_urls:
            raise ValueError('No search results to select from.')
        new_search_results = {}
        if url_to_chunk_score:
            sorted_chunks = []
            for url in select_urls:
                if url in url_to_chunk_score:
                    score_list, chunk_list = [], []
                    chunk_dict = url_to_chunk_score[url]['chunk_dict']
                    score_dict = url_to_chunk_score[url]['scores'] # {index, sccore}
                    for index, score in score_dict.items():
                        idx = extract_int(index)
                        if idx in chunk_dict:
                            chunk_list.append(chunk_dict[idx])
                            score_list.append(score)
                    chunk_score_pairs = [(score, chunk, url) for score, chunk in zip(score_list, chunk_list)]
                    sorted_chunks.extend(sorted(chunk_score_pairs, key=lambda x: x[0], reverse=True))

            sorted_chunks = [(chunk,url) for _, chunk, url in sorted_chunks]
            return sorted_chunks
        
        else:
            titles = []
            for url in select_urls:
                for items in search_results.values():
                    if items['url'] == url:
                        titles.append(items['title'])
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(self._execute, url, title, webpage_cache): url for url, title in zip(select_urls, titles)}
                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result(timeout=15))

                    except TimeoutError:
                        print(f"Timeout: {futures[future]}")
                        webpage_cache.store_failed(url=futures[future], e="TimeoutError")

                    except Exception as e:
                        print(f"Error: {e}")
                        webpage_cache.store_failed(url=futures[future], e=str(e))

            for item in results:
                if item:
                    current_url = item['url']
                    if current_url not in url2id:
                        continue
                    current_id = url2id[current_url]
                    search_results[current_id]['content'] = item['content']
                    search_results[current_id]['date'] = item['date']
                    new_search_results[current_id] = search_results[current_id].copy()
                    new_search_results[current_id].pop('summ')

            return new_search_results

    def _execute(self, url, title, webpage_cache):
        cleaned_url = urllib.parse.unquote(url)
        proxies = {}
        response = self.need_proxy(cleaned_url)
        if response.status_code != 200:
            proxies['http'] = self.proxy
            proxies['https'] = self.proxy
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.get(cleaned_url, proxies=proxies)

        if response.status_code == 200:
            pass

        else:
            serper_api_key = self.api_key
            response = requests.get(f"https://scrape.serper.dev?url={url}&apiKey={serper_api_key}")


        if response.status_code not in [200, 302, 301, 303]:

            webpage_cache.store_failed(url=url, e=f"Status: {response.status_code}")

        # code detection and processing
        if response.encoding and response.encoding.lower() == 'iso-8859-1':
            response.encoding = response.apparent_encoding
        if cleaned_url.lower().endswith('.pdf') or 'pdf' in cleaned_url.lower() or '[PDF]' in title:
            content = self.parse_pdf(response)
        
        elif cleaned_url.lower().endswith(('.docx','.doc')):
            content = self.parse_docx(response)
        
        elif cleaned_url.lower().endswith(('.xls','.xlsx')):
            content = self.parse_excel(response)
        
        else:
            content = self.parse(response)
        try:
            date = find_date(response.content)  
        except Exception as e:
            date = ""
            
        results ={
            "url": url,
            "content": content,
            "date": date
        }

        return results



    def need_proxy(self, url):
        try:
            with requests.Session() as session:
                session.headers.update(headers)
                response = session.get(url, timeout=5) 
        except Exception as e:
            serper_api_key = self.api_key
            response = requests.get(f"https://scrape.serper.dev?url={url}&apiKey={serper_api_key}")
            print(f"using serper.dev:{url} status_code:{response.status_code}")

        return response



    def parse(self, response):

        content_type = response.headers.get('content-type', '')

        if 'text/html' not in content_type:
            print(f"Unsupported content type {content_type} for {response.url}")
            
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
        except Exception:
            soup = BeautifulSoup(response.text, 'html.parser')
            
        article = soup.get_text()
        if not article or len(article)<10:
            return

        article = re.sub(r'\|+', '|', article)

        res = self.chunk_content(article)

        return res


    def parse_pdf(self, response):
        try:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = []
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text(
                            x_tolerance=2,
                            y_tolerance=2,
                            layout=False
                        )
                        if page_text and page_text.strip():
                            text.append(page_text.strip())

                    except PDFTextExtractionNotAllowed:
                        continue
                        
                if not text:
                    return 

                return self.chunk_content("\n".join(text))

        except Exception as e:
            print(f"PDF Process Error: {str(e)} :{response.url}")
            return 

    def parse_docx(self, response):
        """DOCX file parsing method"""
        try:
            doc = Document(BytesIO(response.content))
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            if not text.strip():
                return 
            
            return self.chunk_content(text)
        
        except Exception as e:
            print(f"DOCX Parse Error: {response.url} - {str(e)}")
            return 

    def parse_excel(self, response):
        """Excel file parsing method"""
        try:
            df = pd.read_excel(BytesIO(response.content), sheet_name=None)
            text = ""
            for sheet_name, sheet_data in df.items():
                text += f"\n\n【{sheet_name}】\n"
                text += sheet_data.to_string(index=False)
            
            if not text.strip():
                return 
            
            return self.chunk_content(text)
        
        except Exception as e:
            print(f"Excel parse Error: {response.url} - {str(e)}")
            return 

    def chunk_content(self, text ,chunk_size=512):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}
        return chunk_dict
    