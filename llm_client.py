# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 统一 LLM 客户端
===================================

职责：
1. 封装 Gemini 和 OpenAI 兼容 API 调用
2. 实现自动降级：Gemini → OpenAI
3. 支持用户自定义模型配置

使用方法：
    from llm_client import LLMClient
    
    client = LLMClient()
    response = client.generate("分析这只股票...", system="你是一位股票分析师")
"""

import os
import time
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    text: str              # 响应文本
    model: str             # 使用的模型
    provider: str          # 提供商：gemini / openai
    tokens_used: int = 0   # Token 使用量
    error: str = None      # 错误信息


class LLMClient:
    """
    统一 LLM 客户端
    
    支持的提供商：
    1. Gemini（Google AI）
    2. OpenAI 兼容 API（OpenAI、DeepSeek、Groq 等）
    
    降级策略：
    Gemini 主模型 → Gemini 备用模型 → OpenAI 兼容 API
    """
    
    def __init__(self):
        from config import get_config
        self.config = get_config()
        
        # Gemini 配置
        self.gemini_key = self.config.gemini_api_key
        self.gemini_model = self.config.gemini_model
        self.gemini_fallback = self.config.gemini_model_fallback
        
        # OpenAI 配置
        self.openai_key = self.config.openai_api_key
        self.openai_base = self.config.openai_base_url
        self.openai_model = self.config.openai_model
        
        # 重试配置
        self.max_retries = self.config.gemini_max_retries
        self.retry_delay = self.config.gemini_retry_delay
        self.request_delay = self.config.gemini_request_delay
        
        # 初始化客户端
        self._init_gemini()
        self._init_openai()
        
    def _init_gemini(self):
        """初始化 Gemini 客户端"""
        self.gemini_client = None
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_client = genai
                logger.info(f"Gemini 初始化成功，主模型: {self.gemini_model}")
            except ImportError:
                logger.warning("google-generativeai 未安装")
            except Exception as e:
                logger.error(f"Gemini 初始化失败: {e}")
    
    def _init_openai(self):
        """初始化 OpenAI 兼容客户端"""
        self.openai_client = None
        if self.openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=self.openai_key,
                    base_url=self.openai_base
                )
                logger.info(f"OpenAI 初始化成功，模型: {self.openai_model}, BASE_URL: {self.openai_base}")
            except ImportError:
                logger.warning("openai 库未安装，请运行: pip install openai")
            except Exception as e:
                logger.error(f"OpenAI 初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查是否有可用的 LLM"""
        return bool(self.gemini_client or self.openai_client)
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192
    ) -> LLMResponse:
        """
        生成响应，自动降级
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大输出 token
            
        Returns:
            LLMResponse 对象
        """
        errors = []
        
        # 1. 尝试 Gemini 主模型
        if self.gemini_client:
            response = self._try_gemini(
                prompt, system_prompt, self.gemini_model, 
                temperature, max_tokens
            )
            if response.text:
                return response
            errors.append(f"Gemini({self.gemini_model}): {response.error}")
            
            # 2. 尝试 Gemini 备用模型
            if self.gemini_fallback and self.gemini_fallback != self.gemini_model:
                response = self._try_gemini(
                    prompt, system_prompt, self.gemini_fallback,
                    temperature, max_tokens
                )
                if response.text:
                    return response
                errors.append(f"Gemini({self.gemini_fallback}): {response.error}")
        
        # 3. 降级到 OpenAI 兼容 API
        if self.openai_client:
            response = self._try_openai(
                prompt, system_prompt,
                temperature, max_tokens
            )
            if response.text:
                logger.info(f"✅ 降级到 OpenAI 兼容 API 成功: {self.openai_model}")
                return response
            errors.append(f"OpenAI({self.openai_model}): {response.error}")
        
        # 全部失败
        error_msg = " | ".join(errors) if errors else "无可用 LLM"
        return LLMResponse(
            text="",
            model="none",
            provider="none",
            error=error_msg
        )
    
    def _try_gemini(
        self, 
        prompt: str, 
        system: str, 
        model: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """尝试 Gemini API"""
        for attempt in range(self.max_retries):
            try:
                # 请求间隔
                time.sleep(self.request_delay)
                
                # 构建请求
                full_prompt = f"{system}\n\n{prompt}" if system else prompt
                model_instance = self.gemini_client.GenerativeModel(model)
                
                response = model_instance.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': max_tokens,
                    }
                )
                
                return LLMResponse(
                    text=response.text,
                    model=model,
                    provider='gemini'
                )
                
            except Exception as e:
                error_str = str(e)
                
                # 429 配额限制，等待重试
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Gemini {model} 配额限制，"
                        f"等待 {wait_time}s 重试 ({attempt+1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    # 其他错误，直接返回
                    return LLMResponse(
                        text="",
                        model=model,
                        provider='gemini',
                        error=error_str[:100]
                    )
        
        return LLMResponse(
            text="",
            model=model,
            provider='gemini',
            error=f"重试 {self.max_retries} 次后仍失败"
        )
    
    def _try_openai(
        self, 
        prompt: str, 
        system: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """尝试 OpenAI 兼容 API"""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.openai_model,
                provider='openai',
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
        except Exception as e:
            return LLMResponse(
                text="",
                model=self.openai_model,
                provider='openai',
                error=str(e)[:100]
            )


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端实例"""
    return LLMClient()


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    client = LLMClient()
    
    if client.is_available():
        print("=== LLM 客户端测试 ===")
        
        response = client.generate(
            prompt="请用一句话介绍贵州茅台（600519）",
            system_prompt="你是一位专业的股票分析师"
        )
        
        print(f"提供商: {response.provider}")
        print(f"模型: {response.model}")
        print(f"响应: {response.text[:200]}...")
        
        if response.error:
            print(f"错误: {response.error}")
    else:
        print("未配置任何 LLM API Key")
