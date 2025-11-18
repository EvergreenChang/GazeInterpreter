__all__ = [
    'GeminiLLM',
    'OpenAILLM',
    'DeepseekLLM',
    'LlamaLLM',
    'GemmaLocalLLM'
] 


def __getattr__(name):
    if name == 'GeminiLLM':
        from .gemini import GeminiLLM
        return GeminiLLM
    if name == 'OpenAILLM':
        from .openai import OpenAILLM
        return OpenAILLM
    if name == 'DeepseekLLM':
        from .deepseek import DeepseekLLM
        return DeepseekLLM
    if name == 'LlamaLLM':
        from .llama import LlamaLLM
        return LlamaLLM
    if name == 'GemmaLocalLLM':
        from .gemma_local import GemmaLocalLLM
        return GemmaLocalLLM
    raise AttributeError(f"module 'llm_agents' 没有属性 {name}")


def __dir__():
    return sorted(__all__)
