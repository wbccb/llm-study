# (WIP)RAGFlow源码分析

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

### 文件解析
> api/db/services/task_service.py：实际文件解析

根据不同类型设置单个任务最多处理的页数
- pdf 类型默认处理12页，`doc["parser_id"] == "paper"` 类型的 pdf 一个任务处理22页
- table 类型单个任务处理3000行

文件的解析是根据内容拆分为多个不同任务，通过 `Redis` 消息队列进行暂存，然后进行离线异步处理

### 消息队列消费
> rag/svr/task_executor.py：实际文件解析所产生的队列处理

```python
async def handle_task():
    global DONE_TASKS, FAILED_TASKS
    redis_msg, task = await collect()
    if not task:
        return
    try:
        logging.info(f"handle_task begin for task {json.dumps(task)}")
        CURRENT_TASKS[task["id"]] = copy.deepcopy(task)
        await do_handle_task(task)
        DONE_TASKS += 1
        CURRENT_TASKS.pop(task["id"], None)
        logging.info(f"handle_task done for task {json.dumps(task)}")
    except Exception as e:
        FAILED_TASKS += 1
        CURRENT_TASKS.pop(task["id"], None)
        try:
            err_msg = str(e)
            while isinstance(e, exceptiongroup.ExceptionGroup):
                e = e.exceptions[0]
                err_msg += ' -- ' + str(e)
            set_progress(task["id"], prog=-1, msg=f"[Exception]: {err_msg}")
        except Exception:
            pass
        logging.exception(f"handle_task got exception for task {json.dumps(task)}")
    redis_msg.ack()
```

- 调用 `collect()` 方法从消息队列中获取任务
- 为每一个任务依次调用 `build()` 进行文件的解析
- 调用 `embedding()` 方法进行向量化
- 最终调用 `ELASTICSEARCH.bulk()`


#### build()

根据不同文件类型调用不同的文件解析器

> 比如：rag/app/naive.py 包含了目前主流的 docx、pdf、xlsx、md 等文档的解析

涉及到的解析代码放在 `deepdoc/parser`


## 文件检索->得到检索结果

- 对话的API放在 ``
- 实际处理对话的逻辑代码放在：`api/db/services/dialog_service.py`

文件的检索放在 `rag/nlp/search.py` 的 `search()` 完成，目前实现的是混合搜索
- 文本搜索
- 向量搜索

## 检索结果的重排
> 代码放在：`rag/nlp/search.py` 的 `rerank()`

重排是基于文本匹配得分 + 向量匹配得分混合进行排序，默认文本匹配的权重为 0.3, 向量匹配的权重为 0.7


## 构建大模型输入的prompt

在调用大模型前会调用 `api/db/services/dialog_service.py` 文件中 `message_fit_in()` 根据大模型可用的 token 数量进行过滤

将`检索的内容`，`历史聊天记录`以及`问题`构造为 `prompt`，即可作为大模型的输入


# 参考
1. https://blog.csdn.net/hustyichi/article/details/139162109