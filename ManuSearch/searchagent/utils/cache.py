import os
import json
import requests
import logging
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Set

class WebPageCache:
    """
    Web page content caching mechanism with the following features:
    - Each URL's content is stored as a separate JSON file
    - Uses a mapping file to record URL-to-filename relationships
    - Failed URLs are stored in a separate JSON file
    - Cache failure logs are recorded in a text file
    """
    
    def __init__(self, cache_dir: str = "cache", 
                 url_map_file: str = "url_map.json",
                 failed_urls_file: str = "failed_urls.json",
                 error_log_file: str = "cache_errors.txt",
                 timeout: int = 30):
        """
        Initialize the caching mechanism
        
        Args:
            cache_dir: Cache directory path
            url_map_file: URL mapping filename
            failed_urls_file: Failed URLs storage filename
            error_log_file: Error log filename
            timeout: Request timeout in seconds
        """
        self.cache_dir = cache_dir
        self.content_dir = os.path.join(cache_dir, "content")
        self.url_map_path = os.path.join(cache_dir, url_map_file)
        self.failed_urls_path = os.path.join(cache_dir, failed_urls_file)
        self.error_log_path = os.path.join(cache_dir, error_log_file)
        self.timeout = timeout
        
        # URL to filename mapping
        self.url_map = {}
        
        # Set of failed URLs
        self.failed_urls = {}
        
        # Create necessary directories
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(self.content_dir):
            os.makedirs(self.content_dir)
            
        # Load URL mapping
        if os.path.exists(self.url_map_path):
            try:
                with open(self.url_map_path, 'r', encoding='utf-8') as f:
                    self.url_map = json.load(f)
            except json.JSONDecodeError:
                self._log_error(f"Failed to load URL mapping file: {self.url_map_path}")
                self.url_map = {}
        
        # Load failed URLs
        if os.path.exists(self.failed_urls_path):
            try:
                with open(self.failed_urls_path, 'r', encoding='utf-8') as f:
                    self.failed_urls = json.load(f)
            except json.JSONDecodeError:
                self._log_error(f"Failed to load failed URLs file: {self.failed_urls_path}")
                self.failed_urls = {}
    
    def get_content(self, url: str, force_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Get URL content, reading from cache if available, otherwise fetching from web
        
        Args:
            url: URL to fetch
            force_refresh: Whether to force refresh the cache
            
        Returns:
            Tuple of (success_flag, content)
        """
        # Check if URL is in cache and not forcing refresh
        if url in self.url_map and not force_refresh:
            filename = self.url_map[url]
            file_path = os.path.join(self.content_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"Cache hit: {url}")
                        return True, data
                except Exception as e:
                    self._log_error(f"Failed to read cache file: {file_path}, error: {str(e)}")

        return False, None     

    def store_failed(self, url:str, e:str) -> None:
        """Store a failed URL attempt"""
        if url not in self.failed_urls:
            error_msg = f"URL fetch failed: {url}, error: {e}"
            print(error_msg)
            self._log_error(error_msg)
            
            # Add to failed URLs list
            timestamp = datetime.now().isoformat()
            self.failed_urls[url] = {
                "timestamp": timestamp,
                "error": e
            }
            self._save_failed_urls()
        else:
            error_msg = f"{url} already exists in cache_fail_log"
            print(error_msg)
            self._log_error(error_msg)

    def store_content(self, url: str, data: str) -> bool:
        """
        Manually store URL content to cache
        
        Args:
            url: URL to store
            content: Content to store
            
        Returns:
            Whether storage was successful
        """
        try:
            self._store_url_content(url, data)
            
            # Remove from failed list if previously there
            if url in self.failed_urls:
                del self.failed_urls[url]
                self._save_failed_urls()
                
            print(f"Stored URL content: {url}")
            return True
        except Exception as e:
            error_msg = f"Failed to store URL content: {url}, error: {str(e)}"
            print(error_msg)
            self._log_error(error_msg)
            return False
    
    def _store_url_content(self, url: str, data: str) -> None:
        """
        Store URL content as a separate JSON file
        
        Args:
            url: URL to store
            content: Content to store
        """
        # Generate filename (using URL hash)
        filename = self._get_filename_for_url(url)
        file_path = os.path.join(self.content_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Update URL mapping
        self.url_map[url] = filename
        self._save_url_map()
    
    def _get_filename_for_url(self, url: str) -> str:
        """
        Generate unique filename for URL
        
        Args:
            url: URL
            
        Returns:
            Filename string
        """
        # Use MD5 hash to generate filename
        hash_obj = hashlib.md5(url.encode('utf-8'))
        return f"{hash_obj.hexdigest()}.json"
    
    def _save_url_map(self) -> None:
        """Save URL mapping to file"""
        try:
            with open(self.url_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.url_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log_error(f"Failed to save URL map: {str(e)}")
    
    def _save_failed_urls(self) -> None:
        """Save failed URLs to file"""
        try:
            with open(self.failed_urls_path, 'w', encoding='utf-8') as f:
                json.dump(self.failed_urls, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log_error(f"Failed to save failed URLs list: {str(e)}")
    
    def _log_error(self, message: str) -> None:
        """
        Log error to log file
        
        Args:
            message: Error message
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write to log: {str(e)}")
            print(log_entry)
    
    def get_failed_urls(self) -> Dict[str, Dict]:
        """
        Get list of failed URLs
        
        Returns:
            Dictionary of failed URLs in format {url: {timestamp, error}}
        """
        return self.failed_urls
    
    def retry_failed_urls(self) -> Dict[str, bool]:
        """
        Retry all failed URLs
        
        Returns:
            Dictionary of retry results in format {url: success_status}
        """
        results = {}
        failed_urls_copy = self.failed_urls.copy()
        
        for url in failed_urls_copy:
            success, _ = self.get_content(url, force_refresh=True)
            results[url] = success
        
        return results
    
    def clear_cache(self) -> None:
        """Clear entire cache"""
        # Clear URL mapping
        self.url_map = {}
        self._save_url_map()
        
        # Delete all content files
        for filename in os.listdir(self.content_dir):
            file_path = os.path.join(self.content_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self._log_error(f"Failed to delete cache file: {file_path}, error: {str(e)}")
        
        print("Cache cleared")