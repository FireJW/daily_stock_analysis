# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 财报日历模块
===================================

职责：
1. 获取 A股/美股 未来 T+N 日内的财报发布日期
2. 使用 Tavily 搜索获取最新财报预告信息
3. 格式化财报提醒消息并推送

数据来源：
- A股：东方财富/Tushare 业绩预告
- 美股：Tavily 搜索 + AI 解析
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EarningsEvent:
    """财报事件"""
    stock_code: str      # 股票代码
    stock_name: str      # 股票名称
    market: str          # 市场：CN / US
    earnings_date: str   # 财报日期 YYYY-MM-DD
    earnings_time: str   # 发布时间：盘前/盘后/盘中/待定
    days_until: int      # 距今天数
    report_type: str     # 报告类型：Q1/Q2/Q3/Q4/年报
    source: str          # 数据来源


class EarningsCalendar:
    """财报日历管理器"""
    
    # 美股 AI/机器人 Top 20 热门股票
    DEFAULT_US_STOCKS = [
        'NVDA', 'TSLA', 'AMD', 'MSFT', 'GOOG', 
        'META', 'ISRG', 'SYM', 'TER', 'QCOM',
        'ROK', 'MU', 'ADI', 'ZBRA', 'FANUY',
        'ABBNY', 'SYK', 'PRCT', 'PATH', 'SERV'
    ]
    
    def __init__(self):
        from config import get_config
        self.config = get_config()
        
        # Tavily API Key（取第一个）
        self.tavily_key = (
            self.config.tavily_api_keys[0] 
            if self.config.tavily_api_keys else None
        )
        
        # Tushare token
        self.tushare_token = self.config.tushare_token
        
        # 提前天数
        self.lookahead_days = int(os.getenv('EARNINGS_LOOKAHEAD_DAYS', '7'))
        
    def get_upcoming_earnings(
        self, 
        cn_stocks: List[str] = None, 
        us_stocks: List[str] = None
    ) -> List[EarningsEvent]:
        """
        获取即将发布财报的股票列表
        
        Args:
            cn_stocks: A股代码列表
            us_stocks: 美股代码列表
            
        Returns:
            按日期排序的财报事件列表
        """
        events = []
        today = datetime.now()
        
        # 获取 A股财报
        if cn_stocks:
            cn_events = self._get_cn_earnings(cn_stocks, today)
            events.extend(cn_events)
        
        # 获取美股财报
        if us_stocks:
            us_events = self._get_us_earnings(us_stocks, today)
            events.extend(us_events)
        
        # 按日期排序
        events.sort(key=lambda x: x.days_until)
        
        return events
    
    def _get_cn_earnings(
        self, 
        stock_codes: List[str], 
        today: datetime
    ) -> List[EarningsEvent]:
        """获取 A股财报日期"""
        events = []
        
        for code in stock_codes:
            try:
                # 优先使用 Tushare
                if self.tushare_token:
                    event = self._get_cn_earnings_tushare(code, today)
                    if event:
                        events.append(event)
                        continue
                
                # 备选：使用 Tavily 搜索
                if self.tavily_key:
                    event = self._get_cn_earnings_tavily(code, today)
                    if event:
                        events.append(event)
                        
            except Exception as e:
                logger.warning(f"获取 {code} 财报日期失败: {e}")
                
        return events
    
    def _get_cn_earnings_tushare(
        self, 
        code: str, 
        today: datetime
    ) -> Optional[EarningsEvent]:
        """通过 Tushare 获取 A股财报预告"""
        try:
            import tushare as ts
            pro = ts.pro_api(self.tushare_token)
            
            # 获取业绩预告
            df = pro.forecast(
                ts_code=self._normalize_cn_code(code)
            )
            
            if df.empty:
                return None
            
            row = df.iloc[0]
            ann_date = row.get('ann_date', '')
            
            if not ann_date:
                return None
                
            earnings_date = datetime.strptime(ann_date, '%Y%m%d')
            days_until = (earnings_date - today).days
            
            # 只返回未来 N 天内的
            if 0 <= days_until <= self.lookahead_days:
                return EarningsEvent(
                    stock_code=code,
                    stock_name=row.get('name', code),
                    market='CN',
                    earnings_date=earnings_date.strftime('%Y-%m-%d'),
                    earnings_time='交易日',
                    days_until=days_until,
                    report_type=row.get('end_date', '')[-4:],
                    source='Tushare'
                )
                
        except Exception as e:
            logger.debug(f"Tushare 获取 {code} 失败: {e}")
            
        return None
    
    def _get_cn_earnings_tavily(
        self, 
        code: str, 
        today: datetime
    ) -> Optional[EarningsEvent]:
        """通过 Tavily 搜索 A股财报日期"""
        try:
            query = f"{code} A股 财报 发布日期 2026年"
            result = self._tavily_search(query)
            
            if result:
                # TODO: 使用 LLM 解析搜索结果中的日期
                # 当前返回 None，需要进一步实现
                pass
                
        except Exception as e:
            logger.debug(f"Tavily 搜索 {code} 失败: {e}")
            
        return None
    
    def _get_us_earnings(
        self, 
        symbols: List[str], 
        today: datetime
    ) -> List[EarningsEvent]:
        """获取美股财报日期（通过 Tavily 搜索）"""
        events = []
        
        if not self.tavily_key:
            logger.warning("未配置 TAVILY_API_KEYS，无法获取美股财报日期")
            return events
        
        for symbol in symbols:
            try:
                event = self._get_us_earnings_tavily(symbol, today)
                if event:
                    events.append(event)
                    
            except Exception as e:
                logger.warning(f"获取 {symbol} 财报日期失败: {e}")
                
        return events
    
    def _get_us_earnings_tavily(
        self, 
        symbol: str, 
        today: datetime
    ) -> Optional[EarningsEvent]:
        """通过 Tavily 搜索美股财报日期"""
        try:
            query = f"{symbol} stock earnings report date Q4 2025 2026"
            result = self._tavily_search(query)
            
            if not result:
                return None
            
            # 从搜索结果中提取信息
            # 使用简单的模式匹配，后续可以用 LLM 增强
            content = ' '.join([r.get('content', '') for r in result.get('results', [])])
            
            # TODO: 解析日期（需要 LLM 或正则）
            # 当前仅记录搜索成功
            logger.debug(f"找到 {symbol} 财报信息: {content[:200]}...")
            
            return None
            
        except Exception as e:
            logger.debug(f"Tavily 搜索 {symbol} 失败: {e}")
            
        return None
    
    def _tavily_search(self, query: str) -> Optional[Dict]:
        """执行 Tavily 搜索"""
        if not self.tavily_key:
            return None
            
        try:
            response = requests.post(
                'https://api.tavily.com/search',
                json={
                    'api_key': self.tavily_key,
                    'query': query,
                    'search_depth': 'basic',
                    'max_results': 3
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.warning(f"Tavily 搜索失败: {e}")
            return None
    
    def _normalize_cn_code(self, code: str) -> str:
        """标准化 A股代码格式"""
        code = code.replace('.SH', '').replace('.SZ', '')
        
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        return code
    
    def format_notification(self, events: List[EarningsEvent]) -> str:
        """
        格式化财报提醒消息
        
        Args:
            events: 财报事件列表
            
        Returns:
            格式化的消息字符串
        """
        if not events:
            return ""
        
        lines = [
            "📅 **财报日历提醒**",
            f"⏰ 未来 {self.lookahead_days} 天内有以下财报发布：",
            ""
        ]
        
        # 分组：A股 和 美股
        cn_events = [e for e in events if e.market == 'CN']
        us_events = [e for e in events if e.market == 'US']
        
        if cn_events:
            lines.append("🇨🇳 **A股**")
            for e in cn_events:
                emoji = self._get_urgency_emoji(e.days_until)
                lines.append(f"{emoji} **{e.stock_name}** ({e.stock_code})")
                lines.append(f"   📆 {e.earnings_date} {e.earnings_time}")
                lines.append(f"   ⏳ 还有 **{e.days_until}** 天")
            lines.append("")
        
        if us_events:
            lines.append("🇺🇸 **美股**")
            for e in us_events:
                emoji = self._get_urgency_emoji(e.days_until)
                lines.append(f"{emoji} **{e.stock_name}** ({e.stock_code})")
                lines.append(f"   📆 {e.earnings_date} {e.earnings_time}")
                lines.append(f"   ⏳ 还有 **{e.days_until}** 天")
        
        return "\n".join(lines)
    
    def _get_urgency_emoji(self, days: int) -> str:
        """根据天数返回紧急程度 emoji"""
        if days <= 1:
            return "🔴"  # 紧急
        elif days <= 3:
            return "🟡"  # 注意
        else:
            return "🟢"  # 正常


def get_earnings_calendar() -> EarningsCalendar:
    """获取财报日历实例"""
    return EarningsCalendar()


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    calendar = EarningsCalendar()
    
    # 测试 A股
    cn_stocks = ['600519', '002050', '600096']
    
    # 测试美股 Top 20
    us_stocks = EarningsCalendar.DEFAULT_US_STOCKS[:5]
    
    print("正在获取财报日历...")
    events = calendar.get_upcoming_earnings(cn_stocks, us_stocks)
    
    if events:
        print(calendar.format_notification(events))
    else:
        print("未找到近期财报")
