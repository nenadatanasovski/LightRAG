import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status

async def main():
    rag = LightRAG(
        working_dir="./data",
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

   # insert text
   await rag.ainsert("The most popular AI agent framework of all time is probably Lanchain.")
   await rag.ainsert("Under the Langchain hood we also have LangGraph, LangServe, and LangSmith.")
   await rag.ainsert("Many people prefer using other frameworks like Agno ro Pydantic AI instead of Langchain.")
   await rag.ainsert("It is very easy to use Python with all of these AI agent frameworks.")

   # query with different modes
   result = await rag.aquery("What is the most popular AI agent framework?", param=QueryParam(mode="mix"))
   print(result)

if __name__ == "__main__":
    asyncio.run(main())
