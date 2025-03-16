> 深度源码分析建议结合具体使用场景，可重点关注deepdoc文档解析实现与graphrag图增强检索模块。开发者可通过Docker调试环境（docker/docker-compose.yml）快速搭建测试环境

# RAGFlow源码学习流程

https://github.com/infiniflow/ragflow

核心亮点在于 `深度文档解析（DeepDoc）` 和 `混合检索能力` ，支持多种文件格式（PDF、Word、Excel、PPT等）和结构化/非结构化数据的统一处理

## 架构速成（核心概念+文档解析）
### 理论掌握
* RAG双阶段架构：检索(Retrieval)与生成(Generation)的交互机制
* 文档解析技术栈：PDF/HTML/Markdown的解析差异（重点研究ragflow源码中的deepdoc/模块）
* 文本分块算法：调试recursive_splitter.py并可视化分块效果

### 实战
* 用Echarts实现文档解析过程的可视化（展示原始文本→分块→向量化的数据流）
* 开发分块策略配置界面（滑动窗口/段落分割的参数调节）


## 检索引擎攻坚（向量+关键词混合检索）
### 理论掌握
* 向量索引构建：分析vector_indexer.py的批处理逻辑
* 混合检索策略：跟踪HybridRetrieval.run()的多路召回机制
* 重排序算法：在reranker/模块插入调试日志观察分数变化

### 实战
* 开发检索结果对比面板（并列展示纯向量/关键词/混合检索的效果）
* 实现检索耗时监控组件（记录各阶段时间消耗）


## 业务流程再造（对接现有系统）
### 理论掌握
* 工作流引擎：通过dag_controller.py理解任务编排逻辑
* 大模型适配层：修改llm/adapters/huggingface.py测试不同模型
* 权限控制系统：复写api/auth/模块实现前端权限映射

### 实战
* 将现有前端系统接入RAGFlow（重点改造问答界面和知识库管理）
* 开发实时推理监控面板（展示GPU利用率/响应时间/QPS）