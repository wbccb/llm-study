import{_ as l,c as a,a0 as e,o as t}from"./chunks/framework.OqxnQCTf.js";const d=JSON.parse('{"title":"文档解析技术","description":"","frontmatter":{},"headers":[],"relativePath":"知识库/RAG/RAGFlow源码分析/文档预处理阶段/(TODO)文档解析技术.md","filePath":"知识库/RAG/RAGFlow源码分析/文档预处理阶段/(TODO)文档解析技术.md"}'),o={name:"知识库/RAG/RAGFlow源码分析/文档预处理阶段/(TODO)文档解析技术.md"};function r(n,i,c,u,p,s){return t(),a("div",null,i[0]||(i[0]=[e('<h1 id="文档解析技术" tabindex="-1">文档解析技术 <a class="header-anchor" href="#文档解析技术" aria-label="Permalink to &quot;文档解析技术&quot;">​</a></h1><p>对于来自不同领域、具有不同格式和不同检索要求的大量文档，准确的分析成为一项极具挑战性的任务。DeepDoc 就是为了这个目的而诞生的。到目前为止，DeepDoc 中有两个组成部分：视觉处理和解析器</p><blockquote><p>代码路径：<code>deepdoc/</code></p></blockquote><h2 id="整体逻辑简述" tabindex="-1">整体逻辑简述 <a class="header-anchor" href="#整体逻辑简述" aria-label="Permalink to &quot;整体逻辑简述&quot;">​</a></h2><p><strong>1.结构化解析</strong></p><ul><li>使用 pdfplumber 提取表格数据</li><li>基于 LayoutParser 的文档布局分析</li><li>图像增强处理（扫描件 OCR 优化）</li></ul><p><strong>2.语义分块</strong></p><ul><li>提供预定义模板（conf/chunk_templates）</li><li>动态调整分块粒度（段落/章节/自定义逻辑）</li><li>保留层级关系元数据</li></ul><p><strong>3.增强处理</strong></p><ul><li>关键词提取（2024-11-01 更新）</li><li>关联问题生成（辅助后续检索）</li><li>分块质量评估（通过 rag/quality_check 模块）</li></ul><h2 id="视觉处理" tabindex="-1">视觉处理 <a class="header-anchor" href="#视觉处理" aria-label="Permalink to &quot;视觉处理&quot;">​</a></h2><h3 id="t-ocr-py" tabindex="-1">t_ocr.py <a class="header-anchor" href="#t-ocr-py" aria-label="Permalink to &quot;t_ocr.py&quot;">​</a></h3><p><strong>OCR（Optical Character Recognition，光学字符识别）</strong>：由于许多文档都是以图像形式呈现的，或者至少能够转换为图像，因此OCR是文本提取的一个非常重要、基本，甚至通用的解决方案。</p><h3 id="t-recognizer-py" tabindex="-1">t_recognizer.py <a class="header-anchor" href="#t-recognizer-py" aria-label="Permalink to &quot;t_recognizer.py&quot;">​</a></h3><h4 id="布局识别-layout-recognition" tabindex="-1">布局识别（Layout recognition） <a class="header-anchor" href="#布局识别-layout-recognition" aria-label="Permalink to &quot;布局识别（Layout recognition）&quot;">​</a></h4><p>来自不同领域的文件可能有不同的布局，如报纸、杂志、书籍和简历在布局方面是不同的。只有当机器有准确的布局分析时，它才能决定这些文本部分是连续的还是不连续的，或者这个部分需要表结构识别（Table Structure Recognition，TSR）来处理，或者这个部件是一个图形并用这个标题来描述。</p><p>我们有10个基本布局组件，涵盖了大多数情况：</p><ul><li>文本</li><li>标题</li><li>配图</li><li>配图标题</li><li>表格</li><li>表格标题</li><li>页头</li><li>页尾</li><li>参考引用</li><li>公式</li></ul><h4 id="tsr-table-structure-recognition-表结构识别" tabindex="-1">TSR（Table Structure Recognition，表结构识别） <a class="header-anchor" href="#tsr-table-structure-recognition-表结构识别" aria-label="Permalink to &quot;TSR（Table Structure Recognition，表结构识别）&quot;">​</a></h4><p>数据表是一种常用的结构，用于表示包括数字或文本在内的数据。</p><p>表的结构可能非常复杂，比如层次结构标题、跨单元格和投影行标题。</p><p>除了TSR，我们还将内容重新组合成LLM可以很好理解的句子。TSR任务有五个标签：</p><ul><li>列</li><li>行</li><li>列标题</li><li>行标题</li><li>合并单元格</li></ul><h2 id="解析器" tabindex="-1">解析器 <a class="header-anchor" href="#解析器" aria-label="Permalink to &quot;解析器&quot;">​</a></h2><p>PDF、DOCX、EXCEL和PPT四种文档格式都有相应的解析器。最复杂的是PDF解析器，因为PDF具有灵活性。PDF解析器的输出包括：</p><ul><li>在PDF中有自己位置的文本块（页码和矩形位置）。</li><li>带有PDF裁剪图像的表格，以及已经翻译成自然语言句子的内容。</li><li>图中带标题和文字的图。</li></ul>',26)]))}const b=l(o,[["render",r]]);export{d as __pageData,b as default};
