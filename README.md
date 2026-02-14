# Srimad Bhagwad Gita Chatbot (Local Retrieval + Groq LLM)

## What This Project Is
This project is a Retrieval-Augmented Generation (RAG) chatbot for the Bhagavad Gita.

It:
- reads a PDF,
- splits it into chunks,
- creates local vector embeddings (no paid API),
- stores vectors in `local-index.json`,
- retrieves relevant chunks locally and generates final answers with Groq LLM.

Indexing and retrieval are local. Final answer generation uses Groq API.

## Core Idea
This project builds a local index first, then uses it as context for Groq:

1. `rag.ts` triggers indexing.
2. `prepare.ts` builds vectors and writes a local JSON index.
3. `chat.ts` loads the index, retrieves top matches, and sends context to Groq for answer generation.

This gives a low-cost RAG workflow suitable for experimentation and interview demos.

## Architecture

### 1) Ingestion + Indexing (`prepare.ts`)
- Loads the PDF using `PDFLoader`.
- Splits text with `RecursiveCharacterTextSplitter`.
- Creates deterministic local embeddings (`LocalHashEmbeddings`, 1536-dim).
- Persists records to `local-index.json`.

Record shape:
```json
{
  "id": "uuid",
  "values": [0.01, 0.0, ...],
  "metadata": { "...": "..." },
  "text": "chunk text"
}
```

### 2) Index Entry Point (`rag.ts`)
- Calls `indexTheDocument(filePath)`.
- Use this whenever you want to regenerate vectors after changing PDF/document.

### 3) Terminal Chatbot (`chat.ts`)
- Loads `local-index.json`.
- Embeds user query with the same local embedding logic.
- Retrieves top chunks using hybrid ranking:
  - semantic similarity (cosine on vectors),
  - lexical relevance (token overlap + IDF-style weighting).
- Sends top retrieved chunks to Groq (`llama-3.3-70b-versatile`) to generate the final answer.
- Falls back to local extractive answering if Groq call fails.
- Runs in an interactive terminal loop (`exit`/`quit` to stop).

## Project Flow
```text
PDF -> Chunking -> Local Embedding -> local-index.json -> Local Retrieval -> Groq Answer
```

## How To Run Locally

## 1. Install dependencies
```bash
bun install
```

## 2. Pick your input PDF
Edit `rag.ts`:
```ts
const filePath = "./Srimad Bhagwad Gita.pdf";
```
Change to any local PDF path you want to index.

## 3. Add environment variable
Create/update `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 4. Generate vectors (build local index)
```bash
bun run rag.ts
```
Expected output:
```text
Indexed <N> chunks to local store: .../local-index.json
```

## 5. Start chatbot
```bash
bun run chat.ts
```
Then ask questions in terminal:
```text
> what does krishna say about duty?
```

Exit with:
```text
exit
```

## File Structure
- `prepare.ts`: indexing pipeline (PDF -> chunks -> vectors -> JSON).
- `rag.ts`: indexing runner.
- `chat.ts`: interactive chatbot (local retrieval + Groq generation).
- `local-index.json`: generated vector store.

## Notes and Limitations
- Retrieval is local; answer generation requires internet access to Groq API.
- Answer quality depends on chunking, local embeddings, retrieval ranking, and Groq model output.
- `local-index.json` can be large for big PDFs.
- If document content changes, re-run `bun run rag.ts`.

## Environment
- Runtime: Bun (`bun v1.3.9` tested)
- Language: TypeScript

## Optional Improvements
- Add multi-turn conversation memory.
- Add confidence thresholds for uncertain answers.
- Add CLI flags (`--pdf`, `--index-path`, `--top-k`).
- Replace local hash embeddings with stronger local models later (still offline).
