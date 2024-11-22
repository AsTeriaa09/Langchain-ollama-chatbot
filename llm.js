import { Ollama } from "@langchain/ollama";
import { ChatOllama } from "@langchain/ollama";

const model = new ChatOllama({
    model: "llama3.2",  
    temperature: 0.7,
    maxTokens:1000,
    verbose:true,
  });

const response = await model.invoke("hello");

// const response = await model.batch(["hello","how are you?"]);
// const response = await model.stream("write a short poem about ai");
// for await (const chunk of response){
//     console.log(chunk.content);
// }
console.log(response);