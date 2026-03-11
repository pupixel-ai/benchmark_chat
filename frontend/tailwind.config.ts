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
        ink: "#1d1a16",
        sand: "#f3ede4",
        clay: "#d6c1a2",
        ember: "#b5613b",
        pine: "#6f7b6b",
        mist: "#d9d0c4",
        paper: "#f8f5ef",
        cocoa: "#43352a",
        oat: "#ece4d8",
        fog: "#b9aa95"
      },
      fontFamily: {
        display: ["var(--font-newsreader)"],
        sans: ["var(--font-manrope)"],
        mono: ["var(--font-ibm-plex-mono)"]
      },
      boxShadow: {
        card: "0 24px 60px rgba(68, 50, 31, 0.10)"
      }
    }
  },
  plugins: []
};

export default config;
