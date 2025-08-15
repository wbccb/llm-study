import {BaseListChatMessageHistory} from "@langchain/core/chat_history";
import {
  BaseMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
  StoredMessage
} from "@langchain/core/messages";
import path from "path";
import fs from "node:fs";

export class JsonChatHistory extends BaseListChatMessageHistory {
  lc_namespace = ["langchain", "stores", "message"];

  sessionId: string;
  dir: string;

  constructor(sesiongId: string, dir: string) {
    super({
      sesiongId, dir
    });
    this.sessionId = sesiongId;
    this.dir = dir;
  }

  private getFilePath() {
    return path.join(this.dir, `${this.sessionId}.json`);
  }


  private async saveMessagesToFile(messages: BaseMessage[]): Promise<void> {
    const filePath = this.getFilePath();
    const serializedMessages = mapChatMessagesToStoredMessages(messages);
    try {
      fs.writeFileSync(filePath, JSON.stringify(serializedMessages, null, 2), {
        encoding: "utf-8",
      });
    } catch (e) {
      console.error(`Failed to save chat history to ${filePath}`, e);
    }
  }

  async getMessages(): Promise<BaseMessage[]> {
    const filePath = path.join(this.dir, );
    try {
      if(!fs.existsSync(filePath)) {
        this.saveMessagesToFile([]);
        return [];
      }

      const data = fs.readFileSync(filePath, {encoding: "utf-8"});
      const storedMessages = JSON.parse(data) as StoredMessage[];
      return mapStoredMessagesToChatMessages(storedMessages);
    } catch (e) {

    }
  }


  async addMessages(messages: BaseMessage[]): Promise<void> {
    const existMessages = await this.getMessages();
    const newMessages = existMessages.concat(messages);
    await this.saveMessagesToFile(newMessages);
  }

  async addMessage(message: BaseMessage): Promise<void> {
    const existMessages = await this.getMessages();
    existMessages.push(message)
    await this.saveMessagesToFile(existMessages);
  }

  async clear(): Promise<void> {
    const filePath = this.getFilePath();
    fs.unlinkSync(filePath);
  }
}