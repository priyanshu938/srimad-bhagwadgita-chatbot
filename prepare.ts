import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Embeddings } from "@langchain/core/embeddings";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

// Keep dimension stable between indexing and retrieval.
const EMBEDDING_DIMENSION = 1536;

// Small deterministic local embedding model so indexing works fully offline.
class LocalHashEmbeddings extends Embeddings {
    // L2-normalize vectors to make cosine similarity meaningful.
    private normalize(vector: number[]): number[] {
        const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
        if (norm === 0) {
            return vector;
        }

        return vector.map((value) => value / norm);
    }

    private embed(text: string): number[] {
        const vector = new Array<number>(EMBEDDING_DIMENSION).fill(0);

        // Hash each token into a fixed-size vector bucket.
        const tokens = text.toLowerCase().split(/\s+/).filter(Boolean);
        for (const token of tokens) {
            let hash = 0;
            for (let i = 0; i < token.length; i++) {
                hash = (hash * 31 + token.charCodeAt(i)) >>> 0;
            }
            const index = hash % EMBEDDING_DIMENSION;
            vector[index] = (vector[index] ?? 0) + 1;
        }

        return this.normalize(vector);
    }

    async embedDocuments(documents: string[]): Promise<number[][]> {
        return documents.map((doc) => this.embed(doc));
    }

    async embedQuery(document: string): Promise<number[]> {
        return this.embed(document);
    }
}

// Placeholder to preserve the previous "vector store" abstraction.
async function getVectorStore(embeddings: Embeddings) {
    return embeddings;
}

// Shape used for each locally stored vector entry.
type LocalIndexRecord = {
    id: string;
    values: number[];
    metadata: Record<string, unknown>;
    text: string;
};

// Convert chunked documents into vectors and persist them to local-index.json.
async function addDocuments(documents: Array<{ pageContent: string; metadata: Record<string, unknown> }>) {
    if (documents.length === 0) {
        throw new Error("No non-empty chunks available to index.");
    }

    const localEmbeddings = await getVectorStore(new LocalHashEmbeddings({}));
    const vectors = await localEmbeddings.embedDocuments(documents.map((doc) => doc.pageContent));
    if (vectors.length === 0) {
        throw new Error("Embedding step returned 0 vectors.");
    }

    const records: LocalIndexRecord[] = vectors.map((values, index) => ({
        id: randomUUID(),
        values: values ?? [],
        metadata: {
            ...(documents[index]?.metadata || {})
        },
        text: documents[index]?.pageContent || ""
    }));

    // Store index on disk so chat.ts can query it later.
    const outFile = "./local-index.json";
    const resolvedOutFile = path.resolve(outFile);
    await mkdir(path.dirname(resolvedOutFile), { recursive: true });
    await writeFile(resolvedOutFile, JSON.stringify({ createdAt: new Date().toISOString(), records }, null, 2));
    console.log(`Indexed ${records.length} chunks to local store: ${resolvedOutFile}`);
}

// End-to-end indexing pipeline: load PDF -> split -> filter -> embed -> save.
export async function indexTheDocument(filePath: string) {
    const loader = new PDFLoader(filePath, { splitPages: false });
    const doc = await loader.load();
    const pageContent = doc[0]?.pageContent;

    if (!pageContent) {
        throw new Error(`No readable content found in PDF: ${filePath}`);
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100
    });

    // Split into retrieval-friendly chunks.
    const texts = await textSplitter.splitText(pageContent);
    const nonEmptyTexts = texts
        .map((chunk) => chunk.trim())
        .filter((chunk) => chunk.length > 0);

    // Preserve PDF-level metadata on every chunk.
    const documents = nonEmptyTexts.map((chunk) => {
        return {
            pageContent: chunk,
            metadata: (doc[0]?.metadata || {}) as Record<string, unknown>
        };
    });

    if (texts.length > 0 && documents.length === 0) {
        throw new Error("All generated chunks were empty after trimming.");
    }

    await addDocuments(documents);
}
