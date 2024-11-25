import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOllama } from "@langchain/ollama";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

// load data and create vectorStore
const createVectorStore = async () => {
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/docs/how_to/#langchain-expression-language-lcel"
  );
  const docs = await loader.load();

  // transform
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  // embedded data and storing in vectorstore
  const embedding = new OllamaEmbeddings({ model: "llama3.2" });
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embedding
  );
  return vectorStore;
};

// create retrieval chain
const createChain = async () => {
  const model = new ChatOllama({
    model: "llama3.2",
    temperature: 0.3,
    max_retries: 3,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the following context : {context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  // retrieving data
  const retriever = vectorStore.asRetriever({
    k: 1,
  });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a query to loop up in order get relevant information to the conversation",
    ],
  ]);

  const historyRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });
  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever:historyRetriever,
  });
  return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// chat history
const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, How are i help you?"),
  new HumanMessage("My name is Aster"),
  new AIMessage("Hi Aster, How are i help you?"),
  new HumanMessage("what is lcel?"),
  new AIMessage("LCEL is langchain expression language"),
];

const response = await chain.invoke({
  input: "what is it?",
  chat_history: [chatHistory],
});

console.log(response);
