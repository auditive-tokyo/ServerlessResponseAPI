import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["test/**/*.test.ts"],
    setupFiles: ["dotenv/config"],
    coverage: {
      provider: "v8",
      reporter: ["lcov", "text", "text-summary"],
      include: ["lambda_function/**/*.ts"],
      exclude: ["**/*.d.ts"],
    },
  },
});
