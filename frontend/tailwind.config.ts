import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#111318",
        sand: "#f5f0e8",
        clay: "#dfc3a3",
        ember: "#ff6b35",
        pine: "#1d4d4f",
        mist: "#e5e7eb"
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)"],
        mono: ["var(--font-ibm-plex-mono)"]
      },
      boxShadow: {
        card: "0 20px 45px rgba(17, 19, 24, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
