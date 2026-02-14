import { indexTheDocument } from "./prepare";

// Entry point for local indexing.
// Change this path to any PDF you want to index.
const filePath = "./Srimad Bhagwad Gita.pdf";

// Generates ./local-index.json used by chat.ts.
indexTheDocument(filePath);
