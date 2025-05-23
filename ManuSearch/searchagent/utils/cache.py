import os
import json
import requests
import logging
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Set

class WebPageCache:
    """
    网页内容缓存机制，具有以下特点：
    - 每个URL内容存储为单独的JSON文件
    - 使用映射文件记录URL到文件名的对应关系
    - 失败的URL存储在单独的JSON文件中
    - 缓存失败日志记录在文本文件中
    """
    
    def __init__(self, cache_dir: str = "cache", 
                 url_map_file: str = "url_map.json",
                 failed_urls_file: str = "failed_urls.json",
                 error_log_file: str = "cache_errors.txt",
                 timeout: int = 30):
        """
        初始化缓存机制
        
        Args:
            cache_dir: 缓存目录
            url_map_file: URL映射文件名
            failed_urls_file: 失败URL存储文件名
            error_log_file: 错误日志文件名
            timeout: 请求超时时间(秒)
        """
        self.cache_dir = cache_dir
        self.content_dir = os.path.join(cache_dir, "content")
        self.url_map_path = os.path.join(cache_dir, url_map_file)
        self.failed_urls_path = os.path.join(cache_dir, failed_urls_file)
        self.error_log_path = os.path.join(cache_dir, error_log_file)
        self.timeout = timeout
        
        # URL到文件名的映射
        self.url_map = {}
        
        # 失败的URL集合
        self.failed_urls = {}
        
        # 创建必要的目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(self.content_dir):
            os.makedirs(self.content_dir)
            
        # 加载URL映射
        if os.path.exists(self.url_map_path):
            try:
                with open(self.url_map_path, 'r', encoding='utf-8') as f:
                    self.url_map = json.load(f)
            except json.JSONDecodeError:
                self._log_error(f"无法加载URL映射文件: {self.url_map_path}")
                self.url_map = {}
        
        # 加载失败的URL
        if os.path.exists(self.failed_urls_path):
            try:
                with open(self.failed_urls_path, 'r', encoding='utf-8') as f:
                    self.failed_urls = json.load(f)
            except json.JSONDecodeError:
                self._log_error(f"无法加载失败URL文件: {self.failed_urls_path}")
                self.failed_urls = {}
    
    def get_content(self, url: str, force_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        """
        获取URL内容，优先从缓存读取，缓存不存在则从网络获取
        
        Args:
            url: 要获取的URL
            force_refresh: 是否强制刷新缓存
            
        Returns:
            (成功标志, 内容) 元组
        """
        # 检查URL是否在缓存中且不强制刷新
        if url in self.url_map and not force_refresh:
            filename = self.url_map[url]
            file_path = os.path.join(self.content_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"缓存命中: {url}")
                        return True, data
                except Exception as e:
                    self._log_error(f"读取缓存文件失败: {file_path}, 错误: {str(e)}")

        return False, None     

    def store_failed(self, url:str, e:str) -> None:
        if url not in self.failed_urls:
            error_msg = f"获取URL失败: {url}, 错误: {e}"
            print(error_msg)
            self._log_error(error_msg)
            
            # 添加到失败URL列表
            timestamp = datetime.now().isoformat()
            self.failed_urls[url] = {
                "timestamp": timestamp,
                "error": e
            }
            self._save_failed_urls()
        else:
            error_msg = f"{url} 已经存入cache_fail_log"
            print(error_msg)
            self._log_error(error_msg)


    def store_content(self, url: str, data: str) -> bool:
        """
        手动存储URL内容到缓存
        
        Args:
            url: 要存储的URL
            content: 要存储的内容
            
        Returns:
            存储是否成功
        """
        try:
            self._store_url_content(url, data)
            
            # 如果之前在失败列表中，现在移除
            if url in self.failed_urls:
                del self.failed_urls[url]
                self._save_failed_urls()
                
            print(f"已存储URL内容: {url}")
            return True
        except Exception as e:
            error_msg = f"存储URL内容失败: {url}, 错误: {str(e)}"
            print(error_msg)
            self._log_error(error_msg)
            return False
    
    def _store_url_content(self, url: str, data: str) -> None:
        """
        将URL内容存储为单独的JSON文件
        
        Args:
            url: 要存储的URL
            content: 要存储的内容
        """
        # 生成文件名 (使用URL的哈希值)
        filename = self._get_filename_for_url(url)
        file_path = os.path.join(self.content_dir, filename)
        
        # # 存储内容
        # data = {
        #     "url": url,
        #     "content": content,
        #     "timestamp": datetime.now().isoformat()
        # }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 更新URL映射
        self.url_map[url] = filename
        self._save_url_map()
    
    def _get_filename_for_url(self, url: str) -> str:
        """
        为URL生成唯一的文件名
        
        Args:
            url: URL
            
        Returns:
            文件名
        """
        # 使用MD5哈希生成文件名
        hash_obj = hashlib.md5(url.encode('utf-8'))
        return f"{hash_obj.hexdigest()}.json"
    
    def _save_url_map(self) -> None:
        """保存URL映射到文件"""
        try:
            with open(self.url_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.url_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log_error(f"保存URL映射失败: {str(e)}")
    
    def _save_failed_urls(self) -> None:
        """保存失败的URL到文件"""
        try:
            with open(self.failed_urls_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_urls, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log_error(f"保存失败URL列表失败: {str(e)}")
    
    def _log_error(self, message: str) -> None:
        """
        记录错误到日志文件
        
        Args:
            message: 错误信息
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"写入日志失败: {str(e)}")
            print(log_entry)
    
    def get_failed_urls(self) -> Dict[str, Dict]:
        """
        获取失败的URL列表
        
        Returns:
            失败的URL字典，格式为 {url: {timestamp, error}}
        """
        return self.failed_urls
    
    def retry_failed_urls(self) -> Dict[str, bool]:
        """
        重试所有失败的URL
        
        Returns:
            重试结果字典，格式为 {url: 是否成功}
        """
        results = {}
        failed_urls_copy = self.failed_urls.copy()
        
        for url in failed_urls_copy:
            success, _ = self.get_content(url, force_refresh=True)
            results[url] = success
        
        return results
    
    def clear_cache(self) -> None:
        """清空整个缓存"""
        # 清空URL映射
        self.url_map = {}
        self._save_url_map()
        
        # 删除所有内容文件
        for filename in os.listdir(self.content_dir):
            file_path = os.path.join(self.content_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self._log_error(f"删除缓存文件失败: {file_path}, 错误: {str(e)}")
        
        print("缓存已清空")
