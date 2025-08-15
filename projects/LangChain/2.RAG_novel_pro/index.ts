import {ChatPromptTemplate, MessagesPlaceholder} from "@langchain/core/prompts";
import {RunnablePassthrough, RunnableSequence, RunnableWithMessageHistory} from "@langchain/core/runnables";
import {ChatOpenAI, OpenAIEmbeddings} from "@langchain/openai";
import {StringOutputParser} from "@langchain/core/output_parsers";
import {AIMessage, HumanMessage} from "@langchain/core/messages";
import path from "path";
import {FaissStore} from "@langchain/community/vectorstores/faiss";
import {Document} from "@langchain/core/documents";
import {JsonChatHistory} from "./json-chat-history";


const getRagChain = async () => {

// 1. llm改写提问，转化为更加明确的问题
  const rephraseChainPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "给定以下对话和一个后续问题，请将后续问题重述为一个独立的问题。请注意，重述的问题应该包含足够的信息，使得没有看过对话历史的人也能理解。",
    ],
    new MessagesPlaceholder("history"),
    [
      "human",
      "将以下问题重述为一个独立的问题: \n{question}"
    ],
  ]);
  const rephraseChain = RunnableSequence.from([
    rephraseChainPrompt,
    new ChatOpenAI({
      temperature: 0.2,
    }),
    new StringOutputParser(),
  ]);

// const historyMessages = [new HumanMessage(""), new AIMessage("")];
// const question = "";
// const newQuestion = rephraseChain.invoke({
//   history: historyMessages,
//   question: question,
// });

  // 2. 加载之前存储的RAGChain，获取检索管理类retriever，传递第1步转化的问题得到RAG数据
  async function loadVectorStore() {
    const dir = path.join(__dirname, "./db/qiu");
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await FaissStore.load(dir, embeddings);
    return vectorStore;
  }

  const vectorStore = await loadVectorStore();
  const retriever = vectorStore.asRetriever(2);
  const convertDocsToString = (documents: Document[]): string => {
    return documents.map((document) => document.pageContent).join("\n");
  };
  const contextRetrieverChain = RunnableSequence.from([
    (input) => input.new_question,
    retriever,
    convertDocsToString,
  ]);

  // 3. 组合llm改写提问 + RAG检索
  const SYSTEM_TEMPLATE = `
    你是一个熟读刘慈欣的《球状闪电》的终极原着党，精通根据作品原文详细解释和回答问题，你在回答时会引用作品原文。
    并且回答时仅根据原文，尽可能回答用户问题，如果原文中没有相关内容，你可以回答“原文中没有相关内容”，

    以下是原文中跟用户回答相关的内容：
    {context}
  `;
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "现在，你需要基于原文，回答以下问题：\n{new_question}`"]
  ])
  const model = new ChatOpenAI();
  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      new_question: rephraseChain,
    }),
    RunnablePassthrough.assign({
      context: contextRetrieverChain,
    }),
    prompt,
    model,
    new StringOutputParser(),
  ]);


  // 4. 为RagChain增加聊天记录的功能
  const chatHistoryDIr = path.join(__dirname, "./chat_data");
  const ragChainWithHistory = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: (sessionId) => new JsonChatHistory({sessionId, dir: chatHistoryDIr}),
    historyMessagesKey: "history",
    inputMessagesKey: "question",
  });

  return ragChainWithHistory;
}


async function run() {
  const ragChain = await getRagChain();
  const res = await ragChain.invoke({
      question: ""
    }, {
      configurable: {sessionId: "test"},
    }
  );
  console.log(res);
}
