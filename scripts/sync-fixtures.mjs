// Copy mock fixtures into web/public/data so the UI can run before real data exists.
// Usage (from repo root or web/):  node scripts/sync-fixtures.mjs   (or: npm run sync-fixtures)
// Real exports from the pipeline (python -m pipeline.export) overwrite these files.
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "contracts", "fixtures");
const dst = join(root, "web", "public", "data");
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log(`Copied fixtures -> ${dst}`);
