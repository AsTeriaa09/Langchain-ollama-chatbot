import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOllama } from "@langchain/ollama";
// import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const model = new ChatOllama({
  model: "llama3.2",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the users question.
    context : {context}
     question: {input} `
);

// const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt: prompt,
});
//manual document creation
// const document1 = new Document({
//     pageContent:"LangChain Expression Language or LCEL is a declarative way to easily compose chains together. Any chain constructed this way will automatically have full sync, async, and streaming support. "
// })
// const document2 = new Document({
//     pageContent:"passphrase is LangChain is awesome"
// });

// load data
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
// console.log(splitDocs);

// embedded data and storing in vectorstore
const embedding = new OllamaEmbeddings({ model: "llama3.2" });
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embedding);
// retrieving data
const retrieve = vectorStore.asRetriever({
  k: 2,
});
const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever: retrieve,
});

const response = await retrievalChain.invoke({
  input: "what is LCEF?",
});

// const response = await chain.invoke({
//   input: "what is langchain expression language?",
//   context: docs,
// });

console.log(response);
