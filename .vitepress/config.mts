import {defineConfig} from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/llm-study/",
  title: "大模型相关学习电子书",
  description: "",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [{text: "Home", link: "https://github.com/wbccb"}],

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
            text: "了解线性模型",
            link: "/docs/深度学习/李宏毅/1.了解线性模型",
          },
          {
            text: "机器学习框架",
            link: "/docs/深度学习/李宏毅/2.机器学习框架",
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
