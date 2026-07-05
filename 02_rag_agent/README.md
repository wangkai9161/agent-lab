# 02 RAG Agent

状态：规划中。

本实验将实现一个中文 RAG Agent，计划覆盖文档加载、文本切分、embedding 生成、向量检索和基于检索上下文的回答。

预期模块：

- `rag/document_loader.py`：加载本地文档。
- `rag/text_splitter.py`：切分长文本。
- `rag/embedding_model.py`：封装 embedding 模型调用。
- `rag/vector_store.py`：管理向量索引和相似度检索。
- `rag/rag_agent.py`：组合检索结果和大模型回答。

当前目录作为后续实验预留位，具体实现会在后续补充。
