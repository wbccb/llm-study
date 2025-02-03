import {defineConfig} from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
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
    ],

    socialLinks: [{icon: "github", link: "https://github.com/wbccb"}],
  },
  rewrites: {
    "/index.md": "/docs/python/1.基础知识.md"
  }
});
