import express from 'express';
import url from 'url';
import axios from 'axios';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';

import { ChatPromptTemplate } from '@langchain/core/prompts';
import {
  RunnableLambda,
  RunnableMap,
  RunnablePassthrough,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';


const app = express();  
app.use(express.json());

let loadedVectorStore;
let retriever;


async function initializeVectorStore() {
  try {
    loadedVectorStore = await HNSWLib.load('data', new OpenAIEmbeddings());
    retriever = loadedVectorStore.asRetriever(1);
    console.log("Vector store loaded successfully.");
  } catch (error) {
    console.error("Error loading vector store:", error);
  }
}

initializeVectorStore();

app.get('/', async (request, response) => {
  try {
    const queryObject = url.parse(request.url, true).query;
    const selectedRag = queryObject.selectedRag || "default rag";
    const question = queryObject.question || "What do you mean?";

    if (!retriever) {
      return response.status(500).json({ error: "Vector store not initialized yet." });
    }
 
    let externalData = "";
    try {
      const apiResponse = await axios.get("https://api.example.com/context");
      externalData = apiResponse.data.content || "No additional context found.";
    } catch (apiError) {
      console.warn("⚠Failed to fetch external context. Proceeding without it.");
    }

    const prompt = ChatPromptTemplate.fromMessages([
      [
        'ai',
        'Answer the question from a rag\'s perspective based ' +
        'only on the following context:\n\n{context}\n\nAdditional info: {externalData}'
      ],
      [
        'ai',
        'You are a rag that answers questions for humans. ' +
        'Responses should include answers from a rag’s perspective ' +
        'and incorporate some rag personality. ' +
        'Each type of rag should have unique responses. ' +
        'Remember to keep it light and fun. ' +
        'Please do not say anything that could be offensive.'
      ],
      ['human', '{question}']
    ]);

    const setupAndRetrieval = RunnableMap.from({
      context: new RunnableLambda({
        func: async (input) => {
          const response = await retriever.invoke(input);
          return response.length > 0 ? response[0].pageContent : "No relevant context found.";
        },
      }).withConfig({ runName: 'contextRetriever' }),
      externalData: new RunnableLambda({
        func: () => externalData,
      }),
      question: new RunnablePassthrough(),
    });

    const model = new ChatOpenAI({});
    const outputParser = new StringOutputParser();

    const chain = setupAndRetrieval.pipe(prompt).pipe(model).pipe(outputParser);

    response.setHeader('Content-Type', 'text/plain');
    response.setHeader('Transfer-Encoding', 'chunked');

    const stream = await chain.stream(
      `From the perspective of a ${selectedRag}, can you tell me ${question}?`
    );

    for await (const chunk of stream) {
      if (chunk) {
        response.write(chunk);
      }
    }

    response.end();
  } catch (error) {
    console.error("Error processing request:", error);
    response.status(500).json({ error: "Internal server error" });
  }
});

app.listen(3000, () => {
  console.log(`Server is running on port 3000`);
});
