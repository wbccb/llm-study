# 构建chunks


## chunker.chunk()构建chunks逻辑详细分析

在上面的分析中，我们知道  `build_chunks()` -> `chunker.chunk()` 中打印 `Start to parse`

> 而 `chunker.chunk()` 具体执行了什么内容呢？


`chunker.chunk()`主要根据扩展名选择对应的解析器进行chunk的构建，包括：
- DOCX文件
- PDF文件
- EXCEL文件（CSV/XLSX）
- 纯文本文件（TXT/代码文件）
- Markdown/HTML/JSON文件
- 其它格式：DOC
- 未支持的格式，抛出错误
----------------

接下来我们针对 `PDF文件` 的解析展开分析

> 针对打印信息进行小点的划分

先获取配置数据，得到 `PDF解析策略`：
- DeepDOC
- Plain Text
- 其它
```python
layout_recognizer = parser_config.get("layout_recognize", "DeepDOC")
```

### DeepDOC模式

使用可视化模型提取
- sections段落
- tables表格
- figures图表
```python
pdf_parser = Pdf()
sections, tables, figures = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback, separate_tables_figures=True)
```

------------------

而 `Pdf()` 以及 `pdf_parser()`，如下面代码所示，会依次执行
- OCR started
- Layout analysis
- Table analysis
- Text merged

```python
class Pdf(PdfParser):
    def __init__(self):
        super().__init__()

    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None, separate_tables_figures=False):
        start = timer()
        first_start = start
        callback(msg="OCR started")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR finished ({:.2f}s)".format(timer() - start))
        logging.info("OCR({}~{}): {:.2f}s".format(from_page, to_page, timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._table_transformer_job(zoomin)
        callback(0.65, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        callback(0.67, "Text merged ({:.2f}s)".format(timer() - start))

        if separate_tables_figures:
            tbls, figures = self._extract_table_figure(True, zoomin, True, True, True)
            self._concat_downward()
            logging.info("layouts cost: {}s".format(timer() - first_start))
            return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls, figures
        else:
            tbls = self._extract_table_figure(True, zoomin, True, True)
            # self._naive_vertical_merge()
            self._concat_downward()
            # self._filter_forpages()
            logging.info("layouts cost: {}s".format(timer() - first_start))
            return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls
```
------------------

然后再使用可视化图表模型对 `figures图表` 进行再度提取存入到 `tables` 中
```python
pdf_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures, **kwargs)
boosted_figures = pdf_vision_parser(callback=callback)
tables.extend(boosted_figures)
```

然后再对 `tables表格` 数据进行处理
- 空值过滤
- 单行表格处理
- 多行表格批处理

对表格数据进行结构化处理，转化为适合机器学习模型处理的Token化文档，同时保留多模态信息（如图像、位置坐标等）
- `tokenize`：HTML标签清理、rag_tokenizer.tokenize进行粗粒度分词、rag_tokenizer.fine_grained_tokenize进行细粒度分词
- `d["image"]`：图像绑定，支持跨模态检索（图文联合检索）
- `add_positions(d, poss)`：记录单元格位置，可用于高亮布局或者布局还原

```python
res = tokenize_table(tables, doc, is_english)

def tokenize_table(tbls, doc, eng, batch_size=10):
    res = []
    # add tables
    for (img, rows), poss in tbls:
        if not rows:
            continue
        if isinstance(rows, str):
            d = copy.deepcopy(doc)
            tokenize(d, rows, eng)
            d["content_with_weight"] = rows
            if img:
                d["image"] = img
            if poss:
                add_positions(d, poss)
            res.append(d)
            continue
        de = "; " if eng else "； "
        for i in range(0, len(rows), batch_size):
            d = copy.deepcopy(doc)
            r = de.join(rows[i:i + batch_size])
            tokenize(d, r, eng)
            d["image"] = img
            add_positions(d, poss)
            res.append(d)
    return res
    
def tokenize(d, t, eng):
    d["content_with_weight"] = t
    t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    d["content_ltks"] = rag_tokenizer.tokenize(t)
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
```

最终打印出
- Finish parsing
  结束解析


### Plain Text模式

仅仅提取纯文本，忽略布局

```python
if layout_recognizer == "Plain Text":
    pdf_parser = PlainParser()
sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback)
res = tokenize_table(tables, doc, is_english)
callback(0.8, "Finish parsing.")
```

### 其它模式

依赖视觉模型解析PDF内容，然后使用`tokenize_table()`解析得到的 `tables` 数据形成 `tokens`

> `sections` 等待下一步的 `naive_merge()` 进行合并


```python
vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT, llm_name=layout_recognizer, lang=lang)
pdf_parser = VisionParser(vision_model=vision_model, **kwargs)
sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback)
res = tokenize_table(tables, doc, is_english)
callback(0.8, "Finish parsing.")
```



### naive_merge

经过上面三种解析策略（任选一种）解析后，我们可以得到对应的 `sections` 数据，然后调用 `naive_merge` 合并内容

如下面代码所示，根据 `chunk_token_num` 进行多个 `section` 的合并，力求整体大小不超过 `chunk_token_num`
> 将多个短的 `section` 合并为符合长度要求的文本块


```python
def naive_merge(sections, chunk_token_num=128, delimiter="\n。；！？"):
    if not sections:
        return []
    if isinstance(sections[0], type("")):
        sections = [(s, "") for s in sections]
    cks = [""]
    tk_nums = [0]

    def add_chunk(t, pos):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if not pos:
            pos = ""
        if tnum < 8:
            pos = ""
        # Ensure that the length of the merged chunk does not exceed chunk_token_num  
        if tk_nums[-1] > chunk_token_num:

            if t.find(pos) < 0:
                t += pos
            cks.append(t)
            tk_nums.append(tnum)
        else:
            if cks[-1].find(pos) < 0:
                t += pos
            cks[-1] += t
            tk_nums[-1] += tnum

    for sec, pos in sections:
        add_chunk(sec, pos)

    return cks
```


### tokenize_chunks

将 `chunks` 转化为 结构化文档，包括
- 调用 `tokenize` 进行语义切分，进行文本块清洗与分词
- 记录 `chunk` 在文档中的物理位置（支持 PDF 解析或者默认索引）
- 通过 `pdf_parser.crop()` 提取与文本关联的图像区域的位置

```python
res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser))

def tokenize_chunks(chunks, doc, eng, pdf_parser=None):
    res = []
    # wrap up as es documents
    for ii, ck in enumerate(chunks):
        if len(ck.strip()) == 0:
            continue
        logging.debug("-- {}".format(ck))
        d = copy.deepcopy(doc)
        if pdf_parser:
            try:
                d["image"], poss = pdf_parser.crop(ck, need_position=True)
                add_positions(d, poss)
                ck = pdf_parser.remove_tag(ck)
            except NotImplementedError:
                pass
        else:
            add_positions(d, [[ii]*5])
        tokenize(d, ck, eng)
        res.append(d)
    return res

```


## 视觉模型逻辑分析

`chunker.chunk()`主要根据扩展名选择对应的解析器进行chunk的构建，包括：
- DOCX文件
- PDF文件
- EXCEL文件（CSV/XLSX）
- 纯文本文件（TXT/代码文件）
- Markdown/HTML/JSON文件
- 其它格式：DOC
- 未支持的格式，抛出错误

针对 PDF，会调用 `Pdf()` 以及 `pdf_parser()`

```python
pdf_parser = Pdf()
sections, tables, figures = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback, separate_tables_figures=True)
```

 `Pdf()` 以及 `pdf_parser()`，如下面代码所示，会依次执行
- OCR started
- Layout analysis
- Table analysis
- Text merged

```python
class Pdf(PdfParser):
    def __init__(self):
        super().__init__()

    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None, separate_tables_figures=False):
        start = timer()
        first_start = start
        callback(msg="OCR started")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR finished ({:.2f}s)".format(timer() - start))
        logging.info("OCR({}~{}): {:.2f}s".format(from_page, to_page, timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._table_transformer_job(zoomin)
        callback(0.65, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        callback(0.67, "Text merged ({:.2f}s)".format(timer() - start))

        if separate_tables_figures:
            tbls, figures = self._extract_table_figure(True, zoomin, True, True, True)
            self._concat_downward()
            logging.info("layouts cost: {}s".format(timer() - first_start))
            return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls, figures
        else:
            tbls = self._extract_table_figure(True, zoomin, True, True)
            # self._naive_vertical_merge()
            self._concat_downward()
            # self._filter_forpages()
            logging.info("layouts cost: {}s".format(timer() - first_start))
            return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls
```

### OCR started
### Layout analysis
### Table analysis
### Text merged
