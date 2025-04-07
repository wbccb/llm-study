import {defineConfig} from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  srcDir: "docs",
  base: "/llm-study/",
  title: "大模型相关学习电子书",
  description: "",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [{text: "Home", link: "https://github.com/wbccb/llm-study"}],

    sidebar: [
      {
        text: "python",
        items: [
          {text: "基础知识", link: "/docs/python/1.基础知识.md"},
        ],
      },
      {
        text: "深度学习",
        items: [
          {text: "基本概念", link: "/docs/深度学习/基本概念.md"},
          {text: "DeepSeek", link: "/docs/深度学习/主流大模型原理/DeepSeek.md"},
        ],
      },
      {
        text: "深度学习-李宏毅",
        items: [
          {
            text: "2021年 & 2022年",
            items: [
              {
                text: "了解线性模型",
                link: "docs/深度学习/李宏毅/2021年&2022年/1.了解线性模型.md",
              },
              {
                text: "机器学习框架",
                link: "docs/深度学习/李宏毅/2021年&2022年/2.机器学习框架.md",
              },
            ]
          },
          {
            text: "2023年&2024年",
            items: [
              {
                text: "chatGPT",
                link: "docs/深度学习/李宏毅/2023年&2024年/1.chatGPT.md",
              },
              {
                text: "生成式AI",
                link: "docs/深度学习/李宏毅/2023年&2024年/2.生成式AI.md",
              },
              {
                text: "不训练模型=>强化模型",
                link: "docs/深度学习/李宏毅/2023年&2024年/3.不训练模型=>强化模型.md",
              },
              {
                text: "训练模型步骤",
                link: "docs/深度学习/李宏毅/2023年&2024年/4.训练模型步骤.md",
              },
              {
                text: "Transformer",
                link: "docs/深度学习/李宏毅/2023年&2024年/5.Transformer.md",
              },
              {
                text: "评估模型能力&模型的安全性",
                link: "docs/深度学习/李宏毅/2023年&2024年/6.评估模型能力&模型的安全性.md",
              },
              {
                text: "生成策略",
                link: "docs/深度学习/李宏毅/2023年&2024年/7.生成策略.md",
              },
              {
                text: "Video相关的生成式AI技术",
                link: "docs/深度学习/李宏毅/2023年&2024年/8.Video相关的生成式AI技术.md",
              },
            ]
          },
        ],
      },
      {
        text: "知识库",
        items: [
          {text: "NLP+大模型=>问答", link: "docs/知识库/NLP+大模型=>问答.md"},
          {
            text: "RAGFlow源码分析", items: [
              {
                text: "文件上传&解析整体流程",
                link: "docs/知识库/RAG/RAGFlow源码分析/文档预处理阶段/文件上传&解析整体流程.md",
              },
              {
                text: "混合检索策略",
                link: "docs/知识库/RAG/RAGFlow源码分析/检索阶段/混合检索策略.md",
              },
            ]
          },
        ],
      },
    ],

    socialLinks: [{icon: "github", link: "https://github.com/wbccb"}],
  },
  rewrites: {
    "/index.md": "/docs/python/1.基础知识.md"
  }
});
