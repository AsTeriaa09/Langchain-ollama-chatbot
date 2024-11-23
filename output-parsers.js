import { ChatOllama } from "@langchain/ollama";
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { z } from "zod";
import { StructuredOutputParser } from "langchain/output_parsers";

const model = new ChatOllama({
  model: "llama3.2",
  temperature: 0.7,
});

async function callStringOutputParser() {
  const prompt = PromptTemplate.fromTemplate(
    "you are a comedian. tell a joke on the following word: {input}\n"
  );
  // console.log(await prompt.format({ input : "duck" }));
  const OutputParser = new StringOutputParser();

  const chain = prompt.pipe(model).pipe(OutputParser);
  return await chain.invoke({
    input: "programming",
  });
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromMessages([
    {
      role: "system",
      content:
        "Provide 5 synonyms, separated by commas, for a word that the user will provide.",
    },
    { role: "user", content: "{input}" },
  ]);
  // console.log(await prompt.format({ input : "duck" }));
  const OutputParser = new CommaSeparatedListOutputParser();

  const chain = prompt.pipe(model).pipe(OutputParser);
  return await chain.invoke({
    input: "happy",
  });
}

async function callStructureParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Extract information from the following phrase. formatting instrustions : {format_instructions} Phrase : {phrase}`
  );

  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "name of the person",
    age: "age of the person",
  });

  const chain = prompt.pipe(model).pipe(outputParser);
  return await chain.invoke({
    phrase: "max is 30",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

async function callZodParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Extract information from the following phrase. formatting instrustions : {format_instructions} Phrase : {phrase}`
  );

  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("name of recipe"),
      ingredients: z.array(z.string()).describe("ingredients"),
    })
  );

  const chain = prompt.pipe(model).pipe(outputParser);
  return await chain.invoke({
    phrase: "the ingredients for pudding are milk,flour,eggs.",
    format_instructions: outputParser.getFormatInstructions(),
  });
}

// const response = await callStringOutputParser();
// const response = await callListOutputParser();
// const response = await callStructureParser();
const response = await callZodParser();

console.log(response);
