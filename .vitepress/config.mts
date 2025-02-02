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
