# RAGFlow源码分析

```bash
├── agent/             # 智能体模块（对话流控制）
├── api/               # RESTful API接口层
├── deepdoc/           # 深度文档解析核心（OCR/表格/图像处理）
├── graphrag/          # 图增强RAG实现（知识图谱融合）
├── rag/               # RAG核心逻辑（检索/重排序/生成）
├── docker/            # 容器化部署配置
├── sdk/python/        # Python SDK开发工具包
├── web/               # 前端交互界面
└── 配置文件类（.env, pyproject.toml等）
```

- 多模态文件解析：`deepdoc/`
- 增强检索层：`rag/retrieval/`
- LLM适配框架：`rag/generation`


## 文件上传 & 解析

> api/db/services/task_service.py：实际文件解析

根据不同类型设置单个任务最多处理的页数
- pdf 类型默认处理12页，`doc["parser_id"] == "paper"` 类型的 pdf 一个任务处理22页
- table 类型单个任务处理3000行



