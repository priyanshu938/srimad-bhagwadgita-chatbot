import { readFile } from "node:fs/promises";
import path from "node:path";
import { stdin as input, stdout as output } from "node:process";
import { createInterface } from "node:readline/promises";

// Must match prepare.ts embedding size.
const EMBEDDING_DIMENSION = 1536;
// Number of chunks to retrieve per question.
const TOP_K = 5;
// Max number of sentences to use in final answer text.
const MAX_SENTENCES = 4;
// Common words ignored in lexical scoring.
const STOPWORDS = new Set([
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "for", "by",
    "with", "as", "at", "from", "is", "are", "was", "were", "be", "being", "been", "it", "this",
    "that", "these", "those", "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "my", "your", "me", "him", "them", "do", "does", "did", "can", "could", "should", "would", "will"
]);

type LocalIndexRecord = {
    id: string;
    values: number[];
    metadata: Record<string, unknown>;
    text: string;
};

type LocalIndexFile = {
    createdAt: string;
    records: LocalIndexRecord[];
};

// Retrieval scores tracked for debugging/tuning.
type RankedMatch = {
    record: LocalIndexRecord;
    semanticScore: number;
    lexicalScore: number;
    score: number;
};

// Precomputed lexical structures for fast ranking at query time.
type IndexedRecord = {
    record: LocalIndexRecord;
    tokens: string[];
    tokenSet: Set<string>;
    termFrequency: Map<string, number>;
};

// L2 normalization supports cosine-like similarity.
function normalize(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (norm === 0) {
        return vector;
    }
    return vector.map((value) => value / norm);
}

// Same deterministic embedding logic used during indexing.
function embed(text: string): number[] {
    const vector = new Array<number>(EMBEDDING_DIMENSION).fill(0);
    const tokens = text.toLowerCase().split(/\s+/).filter(Boolean);

    for (const token of tokens) {
        let hash = 0;
        for (let i = 0; i < token.length; i++) {
            hash = (hash * 31 + token.charCodeAt(i)) >>> 0;
        }
        const index = hash % EMBEDDING_DIMENSION;
        vector[index] = (vector[index] ?? 0) + 1;
    }

    return normalize(vector);
}

// Lowercase + split + remove stopwords.
function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .split(/\W+/)
        .map((token) => token.trim())
        .filter((token) => token.length > 2 && !STOPWORDS.has(token));
}

// Dot product after normalization = cosine similarity.
function cosineSimilarity(a: number[], b: number[]): number {
    const length = Math.min(a.length, b.length);
    if (length === 0) {
        return 0;
    }

    let dot = 0;
    for (let i = 0; i < length; i++) {
        dot += (a[i] ?? 0) * (b[i] ?? 0);
    }
    return dot;
}

// Prepare per-document token stats once when app starts.
function buildIndexedRecords(records: LocalIndexRecord[]): IndexedRecord[] {
    return records.map((record) => {
        const tokens = tokenize(record.text);
        const termFrequency = new Map<string, number>();
        for (const token of tokens) {
            termFrequency.set(token, (termFrequency.get(token) ?? 0) + 1);
        }
        return {
            record,
            tokens,
            tokenSet: new Set(tokens),
            termFrequency
        };
    });
}

// Inverse document frequency for lexical weighting.
function buildIdfMap(records: IndexedRecord[]): Map<string, number> {
    const documentFrequency = new Map<string, number>();
    for (const record of records) {
        for (const token of record.tokenSet) {
            documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
        }
    }

    const totalDocs = records.length;
    const idf = new Map<string, number>();
    for (const [token, df] of documentFrequency.entries()) {
        idf.set(token, Math.log((totalDocs + 1) / (df + 1)) + 1);
    }

    return idf;
}

// Term-match score normalized to [0,1].
function lexicalScore(queryTokens: string[], record: IndexedRecord, idfMap: Map<string, number>): number {
    if (queryTokens.length === 0) {
        return 0;
    }

    let hitScore = 0;
    let maxPossible = 0;
    for (const token of queryTokens) {
        const idf = idfMap.get(token) ?? 0.5;
        maxPossible += idf;
        if (record.termFrequency.has(token)) {
            const tf = record.termFrequency.get(token) ?? 0;
            hitScore += Math.min(1.5, 1 + Math.log(tf)) * idf;
        }
    }

    if (maxPossible === 0) {
        return 0;
    }
    return Math.min(1, hitScore / maxPossible);
}

// Hybrid retrieval: semantic similarity + keyword overlap.
function topMatches(query: string, records: IndexedRecord[], idfMap: Map<string, number>, k: number): RankedMatch[] {
    const queryVector = embed(query);
    const queryTokens = tokenize(query);

    const ranked = records.map((indexed) => {
        const semanticScore = cosineSimilarity(queryVector, indexed.record.values ?? []);
        const keywordScore = lexicalScore(queryTokens, indexed, idfMap);
        const score = semanticScore * 0.75 + keywordScore * 0.25;
        return {
            record: indexed.record,
            semanticScore,
            lexicalScore: keywordScore,
            score
        };
    });

    ranked.sort((a, b) => b.score - a.score);
    return ranked.slice(0, k);
}

// Build final answer from top-ranked sentences in retrieved chunks.
function answerFromMatches(query: string, matches: RankedMatch[]): string {
    if (matches.length === 0) {
        return "I couldn't find relevant context in the local index.";
    }

    const queryTerms = tokenize(query);
    const rankedSentences: Array<{ sentence: string; score: number; source: number }> = [];

    for (const [sourceIndex, match] of matches.entries()) {
        const sentences = match.record.text
            .split(/(?<=[.!?])\s+/)
            .map((s) => s.trim())
            .filter((s) => s.length > 20);

        for (const sentence of sentences) {
            const sentenceTokens = tokenize(sentence);
            const sentenceTokenSet = new Set(sentenceTokens);
            let overlap = 0;
            for (const term of queryTerms) {
                if (sentenceTokenSet.has(term)) {
                    overlap += 1;
                }
            }
            const overlapRatio = queryTerms.length > 0 ? overlap / queryTerms.length : 0;
            const sentenceScore = match.score * 0.7 + overlapRatio * 0.3;
            rankedSentences.push({ sentence, score: sentenceScore, source: sourceIndex + 1 });
        }
    }

    rankedSentences.sort((a, b) => b.score - a.score);
    const used = new Set<string>();
    const selected: string[] = [];

    for (const item of rankedSentences) {
        const key = item.sentence.toLowerCase();
        if (used.has(key)) {
            continue;
        }
        used.add(key);
        selected.push(`${item.sentence} [${item.source}]`);
        if (selected.length >= MAX_SENTENCES) {
            break;
        }
    }

    if (selected.length === 0) {
        const fallback = matches[0]?.record.text.slice(0, 500) ?? "";
        return fallback.length > 0
            ? `Best matching excerpt: ${fallback} [1]`
            : "I found matches, but they were empty.";
    }

    return selected.join(" ");
}

// Load local vector index generated by rag.ts/prepare.ts.
async function loadLocalIndex(): Promise<LocalIndexFile> {
    const indexPath = process.env.LOCAL_VECTOR_STORE_PATH || "./local-index.json";
    const resolvedPath = path.resolve(indexPath);
    const raw = await readFile(resolvedPath, "utf-8");
    const parsed = JSON.parse(raw) as LocalIndexFile;

    if (!Array.isArray(parsed.records) || parsed.records.length === 0) {
        throw new Error(`No records found in ${resolvedPath}. Run \`bun run rag.ts\` first.`);
    }

    return parsed;
}

// Terminal chatbot loop: ask question -> retrieve -> answer.
export async function chat() {
    const index = await loadLocalIndex();
    const indexedRecords = buildIndexedRecords(index.records);
    const idfMap = buildIdfMap(indexedRecords);
    const rl = createInterface({ input, output });

    console.log(`Local Bhagavad Gita chatbot ready. Indexed chunks: ${index.records.length}`);
    console.log("Ask a question. Type 'exit' to quit.");

    while (true) {
        const question = (await rl.question("\n> ")).trim();
        if (!question) {
            continue;
        }
        if (question.toLowerCase() === "exit" || question.toLowerCase() === "quit") {
            break;
        }

        // Retrieve best chunks and synthesize a concise answer.
        const matches = topMatches(question, indexedRecords, idfMap, TOP_K);
        const answer = answerFromMatches(question, matches);

        console.log(`\nAnswer:\n${answer}\n`);
    }

    rl.close();
}

await chat();
