#!/usr/bin/env python3
"""
AI 代码审查脚本
用于 GitHub Actions PR Review 工作流
"""
import os
import subprocess
import traceback



MAX_DIFF_LENGTH = 15000


def get_diff():
    """获取 PR 的代码变更"""
    base_ref = os.environ.get('GITHUB_BASE_REF', 'main')
    result = subprocess.run(
        ['git', 'diff', f'origin/{base_ref}...HEAD', '--', '*.py'],
        capture_output=True, text=True
    )
    diff = result.stdout
    truncated = len(diff) > MAX_DIFF_LENGTH
    return diff[:MAX_DIFF_LENGTH], truncated


def get_changed_files():
    """获取修改的文件列表"""
    base_ref = os.environ.get('GITHUB_BASE_REF', 'main')
    result = subprocess.run(
        ['git', 'diff', '--name-only', f'origin/{base_ref}...HEAD', '--', '*.py'],
        capture_output=True, text=True
    )
    return result.stdout.strip().split('\n') if result.stdout.strip() else []


def build_prompt(diff_content, files, truncated):
    """构建审查提示词"""
    truncate_notice = ""
    if truncated:
        truncate_notice = "\n\n> ⚠️ 注意：由于变更内容过长，diff 已被截断，请基于可见部分进行审查。\n"
    
    # 检测核心文件变更
    core_files = [f for f in files if f in ['main.py', 'config.py', 'analyzer.py', 'notification.py']]
    core_notice = ""
    if core_files:
        core_notice = f"\n\n> 🔔 **核心文件变更**: {', '.join(core_files)}，请重点审查！\n"
    
    return f"""你是一位资深 Python 代码审查专家。请审查以下代码变更，并给出专业的审查意见。

## 修改的文件
{', '.join(files)}{core_notice}{truncate_notice}

## 代码变更 (diff)
```diff
{diff_content}
```

## 审查要求
请从以下维度进行审查，使用中文回复：

1. **🔒 安全性**: 是否存在安全漏洞（如 SQL 注入、敏感信息泄露等）
2. **🐛 潜在 Bug**: 是否有逻辑错误、边界条件未处理、异常未捕获
3. **⚡ 性能**: 是否有性能问题（如不必要的循环、内存泄漏风险）
4. **📖 可读性**: 代码是否清晰易懂，命名是否规范
5. **🏗️ 架构设计**: 是否符合项目架构，有无更好的实现方式

## 输出格式
- 如果代码质量良好，简要说明优点
- 如果发现问题，列出具体问题和改进建议
- 给出总体评价：✅ 建议合入 / ⚠️ 建议修改后合入 / ❌ 需要重大修改

请保持简洁，重点突出。"""


def review_with_gemini(prompt):
    """使用 Gemini API 进行审查"""
    api_key = os.environ.get('GEMINI_API_KEY')
    # 优先使用 GEMINI_MODEL_FALLBACK，如果未设置则使用硬编码的默认值
    model = os.environ.get('GEMINI_MODEL_FALLBACK') or 'gemini-2.5-flash'
    
    if not api_key:
        print("❌ Gemini API Key 未配置（检查 GitHub Secrets: GEMINI_API_KEY）")
        return None
    
    print("Gemini API key configured")
    print(f"🤖 使用模型: {model}")
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        print(f"✅ Gemini ({model}) 审查成功")
        return response.text
    except ImportError as e:
        print(f"❌ Gemini 依赖未安装: {e}")
        print("   请确保安装了 google-genai: pip install google-genai")
        return None
    except Exception as e:
        print(f"❌ Gemini 审查失败: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
        return None


def review_with_openai(prompt):
    """使用 OpenAI 兼容 API 进行审查（备用）"""
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    
    if not api_key:
        print("❌ OpenAI API Key 未配置（检查 GitHub Secrets: OPENAI_API_KEY）")
        return None
    
    print("OpenAI API key configured")
    print(f"🌐 Base URL: {base_url}")
    print(f"🤖 使用模型: {model}")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        print(f"✅ OpenAI 兼容接口 ({model}) 审查成功")
        return response.choices[0].message.content
    except ImportError as e:
        print(f"❌ OpenAI 依赖未安装: {e}")
        print("   请确保安装了 openai: pip install openai")
        return None
    except Exception as e:
        print(f"❌ OpenAI 兼容接口审查失败: {e}")
        traceback.print_exc()
        return None


def ai_review(diff_content, files, truncated):
    """调用 AI 进行代码审查，优先 Gemini，失败后尝试 OpenAI"""
    prompt = build_prompt(diff_content, files, truncated)
    
    # 1. 优先尝试 Gemini
    result = review_with_gemini(prompt)
    if result:
        return result
    
    # 2. Gemini 失败，尝试 OpenAI 兼容接口
    print("尝试使用 OpenAI 兼容接口...")
    result = review_with_openai(prompt)
    if result:
        return result
    
    return None


def main():
    diff, truncated = get_diff()
    files = get_changed_files()
    
    if not diff or not files:
        print("没有 Python 代码变更，跳过 AI 审查")
        summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_file:
            with open(summary_file, 'a') as f:
                f.write("## 🤖 AI 代码审查\n\n✅ 没有 Python 代码变更\n")
        return
    
    print(f"审查文件: {files}")
    if truncated:
        print(f"⚠️ Diff 内容已截断至 {MAX_DIFF_LENGTH} 字符")
    
    review = ai_review(diff, files, truncated)
    
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
    
    if review:
        if summary_file:
            with open(summary_file, 'a') as f:
                f.write(f"## 🤖 AI 代码审查\n\n{review}\n")
        
        with open('ai_review_result.txt', 'w') as f:
            f.write(review)
        
        print("AI 审查完成")
    else:
        print("⚠️ 所有 AI 接口都不可用")
        if summary_file:
            with open(summary_file, 'a') as f:
                f.write("## 🤖 AI 代码审查\n\n⚠️ AI 接口不可用，请检查配置\n")


if __name__ == '__main__':
    main()
