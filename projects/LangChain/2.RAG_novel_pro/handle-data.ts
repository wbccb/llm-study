import {TextLoader} from "langchain/document_loaders/fs/text";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import {OpenAIEmbeddings} from "@langchain/openai";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import path from "path";
import "dotenv/config";

const run = async () => {

  const baseDir = __dirname;

  // 加载文件、切割文件、将切割后的文件转化为向量数据、将向量数据存入本地文件中
  const loader = new TextLoader("./data/qiu.txt");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const splitDocs = await splitter.splitDocuments(docs);
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

  return vectorStore.save(path.join(baseDir, "./db/qiu"));
}


run();
