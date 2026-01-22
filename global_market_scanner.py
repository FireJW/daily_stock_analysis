# -*- coding: utf-8 -*-
"""
===================================
全球市场异动扫描器
===================================

职责：
1. 扫描主要全球指数、商品、外汇（过去24h）
2. 识别异动（涨跌幅 > 2%）
3. 调用搜索服务查询异动原因
4. 调用 LLM 关联 A 股概念
"""

import logging
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import get_config
from search_service import SearchService, SearchResponse
from llm_client import get_llm_client

logger = logging.getLogger(__name__)

@dataclass
class MarketAnomaly:
    """市场异动数据类"""
    symbol: str             # 代码 (e.g., "NG=F")
    name: str               # 名称 (e.g., "天然气")
    price: float            # 当前价格
    change_pct: float       # 涨跌幅 (%)
    reason: str = ""        # 异动原因 (AI 分析)
    related_stocks: List[str] = None  # 关联 A 股 (AI 推荐)
    news: List[Dict] = None # 相关新闻

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'reason': self.reason,
            'related_stocks': self.related_stocks or [],
            'news': self.news or []
        }

class GlobalMarketScanner:
    """全球市场扫描器"""
    
    # 监控列表：主要指数、商品、外汇
    # 使用 Yahoo Finance 代码
    WATCHLIST = {
        # === 指数 ===
        '^GSPC': '标普500',
        '^IXIC': '纳斯达克',
        '^DJI': '道琼斯',
        '^N225': '日经225',
        '^HSI': '恒生指数',
        '^FTSE': '富时100',
        
        # === 商品 Futures ===
        'CL=F': '原油(WTI)',
        'BZ=F': '布伦特原油',
        'GC=F': '黄金',
        'SI=F': '白银',
        'NG=F': '天然气',
        'HG=F': '铜',
        'ZC=F': '玉米',
        'ZS=F': '大豆',
        
        # === 外汇/汇率 ===
        'CNY=X': '美元/人民币',
        'JPY=X': '美元/日元',
        'EURUSD=X': '欧元/美元',
        
        # === 波动率 ===
        '^VIX': 'VIX恐慌指数' 
    }
    
    # 异动阈值 (默认 2%)
    THRESHOLD = 2.0
    
    def __init__(self, search_service: Optional[SearchService] = None):
        self.config = get_config()
        self.search_service = search_service
        self.llm_client = get_llm_client()
        
    def scan_and_analyze(self) -> List[MarketAnomaly]:
        """
        扫描市场并分析异动
        
        Returns:
            异动列表
        """
        logger.info("开始扫描全球市场异动...")
        anomalies = []
        
        # 1. 获取数据
        data_map = self._fetch_global_data()
        
        # 2. 识别异动
        for symbol, info in data_map.items():
            change_pct = info.get('change_pct', 0.0)
            
            # 过滤异动 (绝对值 > 阈值)
            # VIX 指数通常波动大，阈值可以设高一点，这里暂统一
            threshold = self.THRESHOLD
            if symbol == '^VIX':
                threshold = 5.0 # VIX 设置高一点
                
            if abs(change_pct) >= threshold:
                name = self.WATCHLIST.get(symbol, symbol)
                logger.info(f"发现异动: {name} ({change_pct:+.2f}%)")
                
                anomaly = MarketAnomaly(
                    symbol=symbol,
                    name=name,
                    price=info.get('price', 0.0),
                    change_pct=change_pct
                )
                anomalies.append(anomaly)
        
        # 3. 分析异动原因 & 关联 A 股
        if anomalies:
            self._analyze_anomalies(anomalies)
            
        logger.info(f"扫描完成，共发现 {len(anomalies)} 个异动")
        return anomalies

    def _fetch_global_data(self) -> Dict[str, Dict]:
        """
        使用 yfinance 获取数据
        """
        results = {}
        symbols = list(self.WATCHLIST.keys())
        
        try:
            # 批量获取行情 (period='2d' 以便计算涨跌)
            # Yahoo Finance 接口可能会有延迟，取最新价或者 prevClose 计算
            tickers = yf.Tickers(" ".join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    # fast_info 通常比 history 更快
                    info = ticker.fast_info
                    
                    price = info.last_price
                    prev_close = info.previous_close
                    
                    if price and prev_close:
                        change = price - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        results[symbol] = {
                            'price': price,
                            'change_pct': change_pct
                        }
                except Exception as e:
                    logger.debug(f"获取 {symbol} 失败: {e}")
                    
        except Exception as e:
            logger.error(f"批量获取全球行情失败: {e}")
            
        return results

    def _analyze_anomalies(self, anomalies: List[MarketAnomaly]):
        """
        对异动进行深入分析 (搜索 + LLM)
        """
        if not self.llm_client.is_available():
            logger.warning("LLM 不可用，跳过异动深度分析")
            return

        for anomaly in anomalies:
            try:
                # 1. 搜索原因
                news_summary = ""
                if self.search_service:
                    query = f"{anomaly.name} price change reason news today"
                    # 中文搜索可能更好
                    query_cn = f"{anomaly.name} 行情 异动 原因 新闻"
                    
                    search_res = self.search_service.search_stock_news(
                        stock_code="global",
                        stock_name=anomaly.name,
                        max_results=3,
                        focus_keywords=query_cn.split()
                    )
                    
                    if search_res and search_res.results:
                        anomaly.news = [
                            {'title': r.title, 'url': r.url} for r in search_res.results
                        ]
                        news_summary = "\n".join([
                            f"- {r.title}: {r.snippet}" for r in search_res.results
                        ])
                
                # 2. LLM 分析 & 关联
                prompt = f"""
                全球市场监控发现异动：
                资产：{anomaly.name}
                涨跌幅：{anomaly.change_pct:+.2f}%
                当前价格：{anomaly.price}
                
                相关新闻：
                {news_summary}
                
                请完成以下任务：
                1. 用一句话概括异动原因（基于新闻）。
                2. 推荐 3 只最相关的 A 股概念股（代码+名称），并说明理由。
                
                输出格式要求 JSON：
                {{
                    "reason": "异动原因总结",
                    "related_stocks": [
                        "600xxx 股票名: 理由",
                        "000xxx 股票名: 理由",
                        ...
                    ]
                }}
                """
                
                response = self.llm_client.generate(prompt, system_prompt="你是一位金融市场分析师，擅长全球宏观与 A 股联动分析。", temperature=0.5)
                
                if response.text:
                    import json
                    import re
                    
                    # 尝试解析 JSON
                    try:
                        text = response.text
                        # 提取代码块中的 json
                        if "```json" in text:
                            match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                            if match:
                                text = match.group(1)
                        elif "```" in text:
                             match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
                             if match:
                                text = match.group(1)
                                
                        data = json.loads(text)
                        anomaly.reason = data.get("reason", "未找到具体原因")
                        anomaly.related_stocks = data.get("related_stocks", [])
                        
                    except Exception as e:
                        logger.warning(f"解析 LLM 响应失败: {e}, 原始内容: {response.text}")
                        anomaly.reason = response.text[:100] # 降级：直接截取文本
                        
            except Exception as e:
                logger.error(f"分析 {anomaly.name} 失败: {e}")

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    scanner = GlobalMarketScanner()
    # Mock data test or real fetch if implemented
    pass
