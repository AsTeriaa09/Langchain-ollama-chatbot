import { ChatOllama } from "@langchain/ollama";
import { PromptTemplate } from "@langchain/core/prompts";

const model = new ChatOllama({
  model: "llama3.2",
  temperature: 0.7,
});

const prompt = PromptTemplate.fromTemplate(
  "you are a comedian. tell a joke on the following word: {input}\n"
);
// console.log(await prompt.format({ input : "duck" }));

const chain = prompt.pipe(model);
const response = await chain.invoke({
  input: "programming.",
});

console.log(response);
