from vectordb import Memory
from langchain.agents import tool
from functools import lru_cache


def mem_cache(func,memory,top_n=3,multiple_factor=2):
    #这个工具类的目的是把llm tools函数进行封装,1 添加lru_chahe,2添加相似度检索过滤
    func = lru_cache(maxsize=128)(func)
    def wrapper_func(*args, **kwargs):
        kwargs["return_num"] = multiple_factor*top_n
        try:
            ans = func(*args, **kwargs)
        except:
            ans = []
        memory.save(f'{args} + {kwargs.values()} + {ans}',{"args":args}|kwargs)
        try :
            ans = memory.search(f'{args} + {kwargs.values()} + {ans}',top_n=top_n)
            return [item["chunk"] for item in ans]
        except:
            return ans[:top_n]
    wrapper_func.__doc__ = func.__doc__
    wrapper_func.__name__ = func.__name__
    wrapper_func.__dict__.update(func.__dict__)
    wrapper_func = tool(wrapper_func)

    return wrapper_func