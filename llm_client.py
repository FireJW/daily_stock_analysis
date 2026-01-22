# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 统一 LLM 客户端（多服务商版）
===================================

职责：
1. 封装 Gemini 和 多个 OpenAI 兼容 API 调用
2. 实现自动降级：Gemini → OpenAI Provider 1 → Provider 2 → ...
3. 支持用户配置多个 API 服务商（负载均衡/降级）

配置方式（支持逗号分隔多个）：
    OPENAI_API_KEY=sk-deepseek-xxx,sk-openai-xxx
    OPENAI_BASE_URL=https://api.deepseek.com/v1,https://api.openai.com/v1
    OPENAI_MODEL=deepseek-chat,gpt-4o-mini

使用方法：
    from llm_client import LLMClient
    
    client = LLMClient()
    response = client.generate("分析这只股票...", system="你是一位股票分析师")
"""

import os
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    text: str              # 响应文本
    model: str             # 使用的模型
    provider: str          # 提供商名称
    tokens_used: int = 0   # Token 使用量
    error: str = None      # 错误信息


@dataclass
class OpenAIProvider:
    """OpenAI 兼容服务配置"""
    name: str              # 提供商名称（从 URL 提取）
    api_key: str
    base_url: str
    model: str


class LLMClient:
    """
    统一 LLM 客户端（多服务商版）
    
    支持的提供商：
    1. Gemini（Google AI）- 主模型 + 备用模型
    2. 多个 OpenAI 兼容 API（按配置顺序降级）
    
    降级策略：
    Gemini 主模型 → Gemini 备用模型 → OpenAI Provider 1 → Provider 2 → ...
    """
    
    def __init__(self):
        from config import get_config
        self.config = get_config()
        
        # Gemini 配置
        self.gemini_key = self.config.gemini_api_key
        self.gemini_model = self.config.gemini_model
        self.gemini_fallback = self.config.gemini_model_fallback
        
        # 重试配置
        self.max_retries = self.config.gemini_max_retries
        self.retry_delay = self.config.gemini_retry_delay
        self.request_delay = self.config.gemini_request_delay
        
        # 初始化 Gemini
        self._init_gemini()
        
        # 解析多个 OpenAI 提供商
        self.openai_providers = self._parse_openai_providers()
        
        # 日志输出配置信息
        self._log_config()
    
    def _parse_openai_providers(self) -> List[OpenAIProvider]:
        """
        解析多个 OpenAI 兼容服务配置
        
        支持格式：
        OPENAI_API_KEY=key1,key2,key3
        OPENAI_BASE_URL=url1,url2,url3
        OPENAI_MODEL=model1,model2,model3
        """
        providers = []
        
        # 获取原始配置
        keys_str = os.getenv('OPENAI_API_KEY', self.config.openai_api_key or '')
        urls_str = os.getenv('OPENAI_BASE_URL', self.config.openai_base_url or '')
        models_str = os.getenv('OPENAI_MODEL', self.config.openai_model or '')
        
        # 分割
        keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        urls = [u.strip() for u in urls_str.split(',') if u.strip()]
        models = [m.strip() for m in models_str.split(',') if m.strip()]
        
        if not keys:
            return []
        
        # 填充：如果 URL 或 Model 数量不足，复用最后一个
        while len(urls) < len(keys):
            urls.append(urls[-1] if urls else 'https://api.openai.com/v1')
        while len(models) < len(keys):
            models.append(models[-1] if models else 'gpt-4o-mini')
        
        # 创建 Provider 列表
        for i, (key, url, model) in enumerate(zip(keys, urls, models)):
            name = self._extract_provider_name(url, i)
            providers.append(OpenAIProvider(
                name=name,
                api_key=key,
                base_url=url,
                model=model
            ))
        
        return providers
    
    def _extract_provider_name(self, url: str, index: int) -> str:
        """从 URL 提取提供商名称"""
        if 'deepseek' in url.lower():
            return 'DeepSeek'
        elif 'groq' in url.lower():
            return 'Groq'
        elif 'together' in url.lower():
            return 'Together'
        elif 'openai' in url.lower():
            return 'OpenAI'
        elif 'openrouter' in url.lower():
            return 'OpenRouter'
        elif 'moonshot' in url.lower():
            return 'Moonshot'
        elif 'zhipu' in url.lower():
            return 'ZhipuAI'
        else:
            return f'Provider-{index+1}'
    
    def _log_config(self):
        """输出配置信息"""
        logger.info("=== LLM 客户端配置 ===")
        if self.gemini_key:
            logger.info(f"  Gemini: {self.gemini_model} (备用: {self.gemini_fallback})")
        for p in self.openai_providers:
            logger.info(f"  {p.name}: {p.model} @ {p.base_url[:30]}...")
        logger.info(f"  降级策略: Gemini → " + " → ".join([p.name for p in self.openai_providers]))
        
    def _init_gemini(self):
        """初始化 Gemini 客户端"""
        self.gemini_client = None
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_client = genai
                logger.info(f"Gemini 初始化成功")
            except ImportError:
                logger.warning("google-generativeai 未安装")
            except Exception as e:
                logger.error(f"Gemini 初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查是否有可用的 LLM"""
        return bool(self.gemini_client or self.openai_providers)
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192
    ) -> LLMResponse:
        """
        生成响应，自动降级
        
        降级顺序（OpenAI 优先）：
        1. OpenAI Provider 1
        2. OpenAI Provider 2
        3. ...
        4. Gemini 主模型
        5. Gemini 备用模型
        """
        errors = []
        
        # 1. 优先尝试 OpenAI 兼容服务（按配置顺序）
        for provider in self.openai_providers:
            response = self._try_openai_provider(
                provider, prompt, system_prompt,
                temperature, max_tokens
            )
            if response.text:
                logger.info(f"✅ 使用 {provider.name} ({provider.model}) 成功")
                return response
            errors.append(f"{provider.name}({provider.model}): {response.error}")
        
        # 2. 降级到 Gemini 主模型
        if self.gemini_client:
            response = self._try_gemini(
                prompt, system_prompt, self.gemini_model, 
                temperature, max_tokens
            )
            if response.text:
                return response
            errors.append(f"Gemini({self.gemini_model}): {response.error}")
            
            # 3. 尝试 Gemini 备用模型
            if self.gemini_fallback and self.gemini_fallback != self.gemini_model:
                response = self._try_gemini(
                    prompt, system_prompt, self.gemini_fallback,
                    temperature, max_tokens
                )
                if response.text:
                    return response
                errors.append(f"Gemini({self.gemini_fallback}): {response.error}")
        
        # 全部失败
        error_msg = " | ".join(errors) if errors else "无可用 LLM"
        logger.error(f"所有 LLM 均失败: {error_msg}")
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
                time.sleep(self.request_delay)
                
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
                    provider='Gemini'
                )
                
            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Gemini {model} 配额限制，"
                        f"等待 {wait_time}s 重试 ({attempt+1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    return LLMResponse(
                        text="",
                        model=model,
                        provider='Gemini',
                        error=error_str[:100]
                    )
        
        return LLMResponse(
            text="",
            model=model,
            provider='Gemini',
            error=f"重试 {self.max_retries} 次后仍失败"
        )
    
    def _try_openai_provider(
        self,
        provider: OpenAIProvider,
        prompt: str, 
        system: str,
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """尝试单个 OpenAI 兼容服务"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url
            )
            
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=provider.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=provider.model,
                provider=provider.name,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
            
        except ImportError:
            return LLMResponse(
                text="",
                model=provider.model,
                provider=provider.name,
                error="openai 库未安装"
            )
        except Exception as e:
            return LLMResponse(
                text="",
                model=provider.model,
                provider=provider.name,
                error=str(e)[:100]
            )


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端实例"""
    return LLMClient()
