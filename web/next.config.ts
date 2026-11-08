import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // The app imports the shared data contract from ../contracts (types.ts + scoring.ts),
  // so Turbopack's root has to be the repo root, not web/.
  turbopack: {
    root: path.join(__dirname, ".."),
  },
};

export default nextConfig;
