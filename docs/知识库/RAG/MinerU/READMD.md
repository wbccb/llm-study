系统主要包含两个交互的 Docker 容器：
- RAGFlow 容器：负责知识库管理、问答流程和与大模型交互。
- 图片服务器容器：使用 FastAPI 搭建，提供 MinerU 提取出的图片资源的 HTTP 访问。


两个容器通过 Docker 自定义网络 (rag-network) 连接。RAGFlow+MinerU_test.py 脚本使用 MinerU 解析 PDF，将提取的图片存储到映射给图片服务器容器的卷中。脚本随后将 MinerU 输出的 Markdown 中的[IMG::...]占位符替换为完整的
HTML<img>标签（包含指向图片服务器的 HTTP URL），然后将处理后的文本上传到 RAGFlow。


RAGFlow 在生成回答时，如果需要引用图片，会依赖其知识库中已经包含的 HTML<img>标签。