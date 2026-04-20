"""
联网搜索模块
使用搜索引擎 API 获取实时信息
"""
import requests
from typing import List, Dict


class WebSearch:
    """
    联网搜索
    使用 Bing/Google 搜索 API 获取实时信息
    """
    
    def __init__(self, api_key: str = None, engine: str = "bing"):
        """
        初始化搜索引擎
        
        Args:
            api_key: API 密钥（可选）
            engine: 搜索引擎名称（bing/google）
        """
        self.api_key = api_key
        self.engine = engine
        self.session = requests.Session()
        
        # Bing Search API
        self.bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
        
        # 备用方案：使用免费搜索（无需 API 密钥）
        self.use_free_search = api_key is None
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        执行搜索
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表，包含：
                - title: 标题
                - snippet: 摘要
                - link: 链接
                - source: 来源
        """
        if self.api_key and self.engine == "bing":
            return self._bing_search(query, num_results)
        else:
            # 免费搜索方案（使用 mock 数据）
            return self._free_search(query, num_results)
    
    def _bing_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """使用 Bing API 搜索"""
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            params = {
                'q': query,
                'count': num_results,
                'mkt': 'zh-CN',
                'textDecorations': True,
                'textFormat': 'HTML'
            }
            
            response = self.session.get(
                self.bing_endpoint,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('webPages', {}).get('value', []):
                results.append({
                    'title': item.get('name', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('url', ''),
                    'source': 'Bing'
                })
            
            return results
            
        except Exception as e:
            print(f"Bing 搜索失败：{e}")
            return self._free_search(query, num_results)
    
    def _free_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        免费搜索方案（无需 API 密钥）
        使用模拟数据返回搜索结果
        """
        # 这是一个占位实现，实际使用时建议配置 API 密钥
        # 或者使用其他免费搜索 API（如 DuckDuckGo）
        
        print(f"⚠ 使用免费搜索模式（返回示例数据）")
        print(f"  建议配置 Bing Search API 以获取真实结果")
        
        # 返回示例数据
        return [
            {
                'title': f'关于"{query}"的相关信息 - 示例 1',
                'snippet': f'这是关于"{query}"的示例搜索结果。配置 API 密钥后可获取真实搜索结果...',
                'link': 'https://example.com/search1',
                'source': '示例'
            },
            {
                'title': f'"{query}"的最新研究进展 - 示例 2',
                'snippet': f'柑橘实蝇检测技术研究进展，包括多视角图像融合、深度学习等方法...',
                'link': 'https://example.com/search2',
                'source': '示例'
            },
            {
                'title': f'"{query}"的防治方法 - 示例 3',
                'snippet': f'柑橘实蝇的防治方法包括物理防治、化学防治和生物防治等多种手段...',
                'link': 'https://example.com/search3',
                'source': '示例'
            }
        ]
    
    def get_news(self, query: str, num_results: int = 3) -> List[Dict]:
        """
        获取新闻
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量
            
        Returns:
            List[Dict]: 新闻列表
        """
        if self.api_key and self.engine == "bing":
            try:
                headers = {
                    'Ocp-Apim-Subscription-Key': self.api_key
                }
                
                params = {
                    'q': query,
                    'count': num_results,
                    'mkt': 'zh-CN'
                }
                
                response = self.session.get(
                    f"{self.bing_endpoint}/news",
                    headers=headers,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for item in data.get('value', []):
                    results.append({
                        'title': item.get('name', ''),
                        'snippet': item.get('description', ''),
                        'link': item.get('url', ''),
                        'source': item.get('provider', [{}])[0].get('name', '未知'),
                        'date': item.get('datePublished', '')
                    })
                
                return results
                
            except Exception as e:
                print(f"获取新闻失败：{e}")
        
        # 返回示例数据
        return [
            {
                'title': f'柑橘实蝇防治新技术发布',
                'snippet': '近日，农业科研人员开发出新型柑橘实蝇检测技术...',
                'link': 'https://example.com/news1',
                'source': '农业日报',
                'date': '2024-04-10'
            }
        ]


# 全局实例
_web_search = None


def get_web_search(api_key: str = None):
    """获取联网搜索实例（单例模式）"""
    global _web_search
    if _web_search is None:
        from config import BING_SEARCH_API_KEY
        api_key = api_key or BING_SEARCH_API_KEY
        _web_search = WebSearch(api_key)
    return _web_search


if __name__ == "__main__":
    # 测试代码
    search = get_web_search()
    
    query = "柑橘实蝇检测方法"
    print(f"\n🔍 联网搜索：{query}")
    results = search.search(query, num_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   来源：{result['source']}")
        print(f"   链接：{result['link']}")
        print(f"   摘要：{result['snippet'][:100]}...")
