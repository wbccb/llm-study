import socket
from logging import exception

import gradio as gr
import webbrowser
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import os
import logging


# 添加重试次数
requests.adapters.DEFAULT_RETRIES = 3  # 增加重试次数
session = requests.Session()


# LLM->生成Text Embeddings
EMBED_MODEL = SentenceTransformer("all-MIniLM-L6-v2")
# 向量数据库
CHROMA_CLIENT = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
## 向量数据库->获取对应的集合
COLLECTION = CHROMA_CLIENT.get_or_create_collection("rag_docs")

# 修改布局部分，添加一个新的标签页
with gr.Blocks(
        title="本地RAG问答系统",
        css="""
    /* 全局主题变量 */
    :root[data-theme="light"] {
        --text-color: #2c3e50;
        --bg-color: #ffffff;
        --panel-bg: #f8f9fa;
        --border-color: #e9ecef;
        --success-color: #4CAF50;
        --error-color: #f44336;
        --primary-color: #2196F3;
        --secondary-bg: #ffffff;
        --hover-color: #e9ecef;
        --chat-user-bg: #e3f2fd;
        --chat-assistant-bg: #f5f5f5;
    }

    :root[data-theme="dark"] {
        --text-color: #e0e0e0;
        --bg-color: #1a1a1a;
        --panel-bg: #2d2d2d;
        --border-color: #404040;
        --success-color: #81c784;
        --error-color: #e57373;
        --primary-color: #64b5f6;
        --secondary-bg: #2d2d2d;
        --hover-color: #404040;
        --chat-user-bg: #1e3a5f;
        --chat-assistant-bg: #2d2d2d;
    }

    /* 全局样式 */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
        width: 100vw;
        height: 100vh;
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 1% !important;
        color: var(--text-color);
        background-color: var(--bg-color);
        min-height: 100vh;
    }

    /* 确保标签内容撑满 */
    .tabs.svelte-710i53 {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    /* 主题切换按钮 */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background: var(--panel-bg);
        color: var(--text-color);
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .theme-toggle:hover {
        background: var(--hover-color);
    }

    /* 面板样式 */
    .left-panel {
        padding-right: 20px;
        border-right: 1px solid var(--border-color);
        background: var(--bg-color);
        width: 100%;
    }

    .right-panel {
        height: 100vh;
        background: var(--bg-color);
        width: 100%;
    }

    /* 文件列表样式 */
    .file-list {
        margin-top: 10px;
        padding: 12px;
        background: var(--panel-bg);
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 答案框样式 */
    .answer-box {
        min-height: 500px !important;
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 输入框样式 */
    textarea {
        background: var(--panel-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 14px !important;
    }

    /* 按钮样式 */
    button.primary {
        background: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    button.primary:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* 标题和文本样式 */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    .footer-note {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 13px;
        margin-top: 12px;
    }

    /* 加载和进度样式 */
    #loading, .progress-text {
        color: var(--text-color);
    }

    /* 聊天记录样式 */
    .chat-container {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 16px;
        max-height: 80vh;
        height: 80vh !important;
        overflow-y: auto;
        background: var(--bg-color);
    }

    .chat-message {
        padding: 12px 16px;
        margin: 8px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
    }

    .chat-message.user {
        background: var(--chat-user-bg);
        margin-left: 32px;
        border-top-right-radius: 4px;
    }

    .chat-message.assistant {
        background: var(--chat-assistant-bg);
        margin-right: 32px;
        border-top-left-radius: 4px;
    }

    .chat-message .timestamp {
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 4px;
    }

    .chat-message .content {
        white-space: pre-wrap;
    }

    /* 按钮组样式 */
    .button-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }

    .clear-button {
        background: var(--error-color) !important;
    }

    /* API配置提示样式 */
    .api-info {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
    }

    /* 新增: 数据可视化卡片样式 */
    .model-card {
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border-color);
        margin-bottom: 16px;
    }

    .model-card h3 {
        margin-top: 0;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
    }

    .model-item {
        display: flex;
        margin-bottom: 8px;
    }

    .model-item .label {
        flex: 1;
        font-weight: 500;
    }

    .model-item .value {
        flex: 2;
    }

    /* 数据表格样式 */
    .chunk-table {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .chunk-table th, .chunk-table td {
        border: 1px solid var(--border-color);
        padding: 8px;
    }

    .chunk-detail-box {
        min-height: 200px;
        padding: 16px;
        background: var(--panel-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-family: monospace;
        white-space: pre-wrap;
        overflow-y: auto;
    }
    """
) as demo:
    gr.Markdown("# 🧠 智能文档问答系统")

    with gr.Tabs() as tabs:
        # 第一个选项卡：问答对话
        with gr.TabItem("💬 问答对话"):
            with gr.Row(equal_height=True):
                # 左侧操作面板 - 调整比例为合适的大小
                with gr.Column(scale=5, elem_classes="left-panel"):
                    gr.Markdown("## 📂 文档处理区")
                    with gr.Group():
                        file_input = gr.File(
                            label="上传PDF文档",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("🚀 开始处理", variant="primary")
                        upload_status = gr.Textbox(
                            label="处理状态",
                            interactive=False,
                            lines=2
                        )
                        file_list = gr.Textbox(
                            label="已处理文件",
                            interactive=False,
                            lines=3,
                            elem_classes="file-list"
                        )

                    # 将问题输入区移至左侧面板底部
                    gr.Markdown("## ❓ 输入问题")
                    with gr.Group():
                        question_input = gr.Textbox(
                            label="输入问题",
                            lines=3,
                            placeholder="请输入您的问题...",
                            elem_id="question-input"
                        )
                        with gr.Row():
                            # 添加联网开关
                            web_search_checkbox = gr.Checkbox(
                                label="启用联网搜索",
                                value=False,
                                info="打开后将同时搜索网络内容（需配置SERPAPI_KEY）"
                            )

                            # 添加模型选择下拉框
                            model_choice = gr.Dropdown(
                                choices=["ollama", "siliconflow"],
                                value="ollama",
                                label="模型选择",
                                info="选择使用本地模型或云端模型"
                            )

                        with gr.Row():
                            ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button",
                                                  scale=1)

                    # 添加API配置提示信息
                    api_info = gr.HTML(
                        """
                        <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
                            <p>📢 <strong>功能说明：</strong></p>
                            <p>1. <strong>联网搜索</strong>：%s</p>
                            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
                        </div>
                        """
                    )

                # 右侧对话区 - 调整比例
                with gr.Column(scale=7, elem_classes="right-panel"):
                    gr.Markdown("## 📝 对话记录")

                    # 对话记录显示区
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=600,  # 增加高度
                        elem_classes="chat-container",
                        show_label=False
                    )

                    status_display = gr.HTML("", elem_id="status-display")
                    gr.Markdown("""
                    <div class="footer-note">
                        *回答生成可能需要1-2分钟，请耐心等待<br>
                        *支持多轮对话，可基于前文继续提问
                    </div>
                    """)

        # 第二个选项卡：分块可视化
        with gr.TabItem("📊 分块可视化"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 💡 系统模型信息")

                    # 显示系统模型信息卡片
                    models_info = get_system_models_info()
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### 核心模型与技术")

                        for key, value in models_info.items():
                            with gr.Row():
                                gr.Markdown(f"**{key}**:", elem_classes="label")
                                gr.Markdown(f"{value}", elem_classes="value")

                with gr.Column(scale=2):
                    gr.Markdown("## 📄 文档分块统计")
                    refresh_chunks_btn = gr.Button("🔄 刷新分块数据", variant="primary")
                    chunks_status = gr.Markdown("点击按钮查看分块统计")

            # 分块数据表格和详情
            with gr.Row():
                chunks_data = gr.Dataframe(
                    headers=["来源", "序号", "字符数", "分词数", "内容预览"],
                    elem_classes="chunk-table",
                    interactive=False,
                    wrap=True,
                    row_count=(10, "dynamic")
                )

            with gr.Row():
                chunk_detail_text = gr.Textbox(
                    label="分块详情",
                    placeholder="点击表格中的行查看完整内容...",
                    lines=8,
                    elem_classes="chunk-detail-box"
                )

            gr.Markdown("""
            <div class="footer-note">
                * 点击表格中的行可查看该分块的完整内容<br>
                * 分词数表示使用jieba分词后的token数量
            </div>
            """)

    # 调整后的加载提示
    gr.HTML("""
    <div id="loading" style="text-align:center;padding:20px;">
        <h3>🔄 系统初始化中，请稍候...</h3>
    </div>
    """)

    # 进度显示组件调整到左侧面板下方
    with gr.Row(visible=False) as progress_row:
        gr.HTML("""
        <div class="progress-text">
            <span>当前进度：</span>
            <span id="current-step" style="color: #2b6de3;">初始化...</span>
            <span id="progress-percent" style="margin-left:15px;color: #e32b2b;">0%</span>
        </div>
        """)


    # 定义函数处理事件
    def clear_chat_history():
        return None, "对话已清空"


    def process_chat(question, history, enable_web_search, model_choice):
        if history is None:
            history = []

        # 更新模型选择信息的显示
        api_text = """
        <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>功能说明：</strong></p>
            <p>1. <strong>联网搜索</strong>：%s</p>
            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
        </div>
        """ % (
            "已启用" if enable_web_search else "未启用",
            "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
            "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
        )

        # 如果问题为空，不处理
        if not question or question.strip() == "":
            history.append(("", "问题不能为空，请输入有效问题。"))
            return history, "", api_text

        # 添加用户问题到历史
        history.append((question, ""))

        # 创建生成器
        resp_generator = stream_answer(question, enable_web_search, model_choice)

        # 流式更新回答
        for response, status in resp_generator:
            history[-1] = (question, response)
            yield history, "", api_text


    def update_api_info(enable_web_search, model_choice):
        api_text = """
        <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>功能说明：</strong></p>
            <p>1. <strong>联网搜索</strong>：%s</p>
            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
        </div>
        """ % (
            "已启用" if enable_web_search else "未启用",
            "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
            "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
        )
        return api_text


    # 绑定UI事件
    upload_btn.click(
        process_multiple_pdfs,
        inputs=[file_input],
        outputs=[upload_status, file_list],
        show_progress=True
    )

    # 绑定提问按钮
    ask_btn.click(
        process_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input, api_info]
    )

    # 绑定清空按钮
    clear_btn.click(
        clear_chat_history,
        inputs=[],
        outputs=[chatbot, status_display]
    )

    # 当切换联网搜索或模型选择时更新API信息
    web_search_checkbox.change(
        update_api_info,
        inputs=[web_search_checkbox, model_choice],
        outputs=[api_info]
    )

    model_choice.change(
        update_api_info,
        inputs=[web_search_checkbox, model_choice],
        outputs=[api_info]
    )

    # 新增：分块可视化刷新按钮事件
    refresh_chunks_btn.click(
        fn=get_document_chunks,
        outputs=[chunks_data, chunks_status]
    )

    # 新增：分块表格点击事件
    chunks_data.select(
        fn=show_chunk_details,
        inputs=chunks_data,
        outputs=chunk_detail_text
    )

# 修改JavaScript注入部分
demo._js = """
function gradioApp() {
    // 设置默认主题为暗色
    document.documentElement.setAttribute('data-theme', 'dark');

    const observer = new MutationObserver((mutations) => {
        document.getElementById("loading").style.display = "none";
        const progress = document.querySelector('.progress-text');
        if (progress) {
            const percent = document.querySelector('.progress > div')?.innerText || '';
            const step = document.querySelector('.progress-description')?.innerText || '';
            document.getElementById('current-step').innerText = step;
            document.getElementById('progress-percent').innerText = percent;
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
}

function toggleTheme() {
    const root = document.documentElement;
    const currentTheme = root.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', newTheme);
}

// 初始化主题
document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
});
"""


# 文件状态处理管理类
class FileProcessor:
    def __init__(self):
        self.processed_files = {}

    def clear_files(self):
        self.processed_files = {}

    def add_file(self, file_name):
        self.processed_files[file_name] = {
            "status": "等待处理",
            "tiemstamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": 0
        }

    def update_status(self, file_name, status, chunks=None):
        if file_name in self.processed_files:
            self.processed_files[file_name]["status"] = status
            if chunks is not None:
                self.processed_files[file_name]["chunks"] = chunks

    def get_file_list(self):
        # 语法，注意两行之间是没有逗号的
        return [
            f"{fname} | {info['status']}"
            for fname, info in self.processed_files.items()
        ]

file_processor = FileProcessor()


def extract_text(filepath):



# 文档处理流程-PDF解析与分块
def process_multiple_pdfs(files, progress=gr.Progress()):
    # 开始清理旧的数据
    # 处理目前上传的所有数据files
    total_files = len(files)
    processed_results = []
    total_chunks = 0

    for idx, file in enumerate(files, 1):
        try:
            # 可视化展示目前处理文件名称
            file_name = os.path.basename(file)

            # 添加文件到处理器 + 提取该文件内容

            # 对提取的内容进行分割为chunks

            # 使用all-MIniLM-L6-v2将chunks->向量Text Embeddings

            # 将生成的向量以及对应的文档数据存入到chromadb

            # 更新当前UI的处理状态

        except Exception as e:
            error_msg = str(e)
            logging.error(f"处理文件 {file_name}时出错：{error_msg}")
            file_processor.update_status(file_name, f"处理失败: {error_msg}")
            processed_results.append(f"❌ {file_name}: 处理失败 - {error_msg}")

    # 遍历完成，进行信息的总结

    # 更新BM25的索引：可以根据这个索引得到 query与各个文档之间的相关性

    # 获取更新状态后的文件列表（每一个文件都更新了对应的状态)
    file_list = file_processor.get_file_list()

    # 返回处理过程的打印信息
    return "\n".join(processed_results), file_list





def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0


if __name__ == "__main" :
    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)

    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)

    try:
        ollama_check = session.get("http://localhost:11434", timeout=5)
        if ollama_check.status_code != 200:
            print("ollama服务未正常启动")
            exit(1)

        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port,
            server_name="0.0.0.0",
            show_error=True,
            ssl_verify=False,
            height=900
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")