import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { OpenAIEmbeddings } from '@langchain/openai';

async function main() {
  try {
    const loader = new DirectoryLoader('rags', {
      ".txt": (path) => new TextLoader(path),
    });

    console.log("Loading documents...");
    const docs = await loader.load();
    console.log(`Loaded ${docs.length} documents.`);

    const embeddings = new OpenAIEmbeddings();

    console.log("Creating vector store...");
    const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);

    console.log("Saving vector store...");
    await vectorStore.save('data');

    console.log("Vector store saved successfully.");
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
