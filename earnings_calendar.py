# -*- coding: utf-8 -*-
"""
===================================
财报日历模块（最终版）
===================================

功能：
1. 获取 A股/美股 未来 T+7 日内的财报发布日期
2. 格式化财报提醒消息
3. 支持 Tushare 和 Tavily 数据源
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
    stock_code: str
    stock_name: str
    market: str          # CN / US
    earnings_date: str   # YYYY-MM-DD
    earnings_time: str   # 盘前/盘后/盘中/待定
    days_until: int
    report_type: str     # Q1/Q2/Q3/Q4
    source: str


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
        # 尝试从 config 获取配置
        try:
            from config import get_config
            config = get_config()
            self.tavily_key = config.tavily_api_keys[0] if config.tavily_api_keys else None
            self.tushare_token = config.tushare_token
        except:
            # 直接从环境变量获取
            tavily_keys = os.getenv('TAVILY_API_KEYS', '')
            self.tavily_key = tavily_keys.split(',')[0].strip() if tavily_keys else None
            self.tushare_token = os.getenv('TUSHARE_TOKEN')
        
        self.lookahead_days = int(os.getenv('EARNINGS_LOOKAHEAD_DAYS', '7'))
        
    def get_upcoming_earnings(
        self, 
        cn_stocks: List[str] = None, 
        us_stocks: List[str] = None
    ) -> List[EarningsEvent]:
        """获取即将发布财报的股票列表"""
        events = []
        today = datetime.now()
        
        # 过滤空字符串
        cn_stocks = [s.strip() for s in (cn_stocks or []) if s.strip()]
        us_stocks = [s.strip() for s in (us_stocks or []) if s.strip()]
        
        if cn_stocks:
            cn_events = self._get_cn_earnings(cn_stocks, today)
            events.extend(cn_events)
        
        if us_stocks:
            us_events = self._get_us_earnings(us_stocks, today)
            events.extend(us_events)
        
        events.sort(key=lambda x: x.days_until)
        return events
    
    def _get_cn_earnings(self, stock_codes: List[str], today: datetime) -> List[EarningsEvent]:
        """获取 A股财报日期"""
        events = []
        
        for code in stock_codes:
            try:
                if self.tushare_token:
                    event = self._get_cn_earnings_tushare(code, today)
                    if event:
                        events.append(event)
            except Exception as e:
                logger.debug(f"获取 {code} 财报日期失败: {e}")
                
        return events
    
    def _get_cn_earnings_tushare(self, code: str, today: datetime) -> Optional[EarningsEvent]:
        """通过 Tushare 获取 A股财报预告"""
        try:
            import tushare as ts
            pro = ts.pro_api(self.tushare_token)
            
            df = pro.forecast(ts_code=self._normalize_cn_code(code))
            
            if df.empty:
                return None
            
            row = df.iloc[0]
            ann_date = row.get('ann_date', '')
            
            if not ann_date:
                return None
                
            earnings_date = datetime.strptime(str(ann_date), '%Y%m%d')
            days_until = (earnings_date - today).days
            
            if 0 <= days_until <= self.lookahead_days:
                return EarningsEvent(
                    stock_code=code,
                    stock_name=row.get('name', code),
                    market='CN',
                    earnings_date=earnings_date.strftime('%Y-%m-%d'),
                    earnings_time='交易日',
                    days_until=days_until,
                    report_type=str(row.get('end_date', ''))[-4:],
                    source='Tushare'
                )
                
        except ImportError:
            logger.debug("tushare 未安装")
        except Exception as e:
            logger.debug(f"Tushare 获取 {code} 失败: {e}")
            
        return None
    
    def _get_us_earnings(self, symbols: List[str], today: datetime) -> List[EarningsEvent]:
        """获取美股财报日期"""
        # 美股财报日期获取需要 LLM 解析 Tavily 搜索结果
        # 当前版本仅记录日志，后续可扩展
        logger.info(f"美股财报检查: {symbols[:5]}...")
        return []
    
    def _normalize_cn_code(self, code: str) -> str:
        """标准化 A股代码格式"""
        code = code.replace('.SH', '').replace('.SZ', '').strip()
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        return code
    
    def format_notification(self, events: List[EarningsEvent]) -> str:
        """格式化财报提醒消息"""
        if not events:
            return ""
        
        lines = [
            "📅 **财报日历提醒**",
            f"⏰ 未来 {self.lookahead_days} 天内有以下财报发布：",
            ""
        ]
        
        cn_events = [e for e in events if e.market == 'CN']
        us_events = [e for e in events if e.market == 'US']
        
        if cn_events:
            lines.append("🇨🇳 **A股**")
            for e in cn_events:
                emoji = "🔴" if e.days_until <= 1 else "🟡" if e.days_until <= 3 else "🟢"
                lines.append(f"{emoji} **{e.stock_name}** ({e.stock_code})")
                lines.append(f"   📆 {e.earnings_date} {e.earnings_time}")
                lines.append(f"   ⏳ 还有 **{e.days_until}** 天")
            lines.append("")
        
        if us_events:
            lines.append("🇺🇸 **美股**")
            for e in us_events:
                emoji = "🔴" if e.days_until <= 1 else "🟡" if e.days_until <= 3 else "🟢"
                lines.append(f"{emoji} **{e.stock_name}** ({e.stock_code})")
                lines.append(f"   📆 {e.earnings_date} {e.earnings_time}")
                lines.append(f"   ⏳ 还有 **{e.days_until}** 天")
        
        return "\n".join(lines)


def get_earnings_calendar() -> EarningsCalendar:
    """获取财报日历实例"""
    return EarningsCalendar()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    cal = EarningsCalendar()
    cn_stocks = ['600519', '002050', '600096']
    us_stocks = EarningsCalendar.DEFAULT_US_STOCKS[:5]
    
    print("正在获取财报日历...")
    events = cal.get_upcoming_earnings(cn_stocks, us_stocks)
    
    if events:
        print(cal.format_notification(events))
    else:
        print("未找到近期财报")
