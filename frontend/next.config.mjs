const internalApiBase =
  (process.env.INTERNAL_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** @type {import('next').NextConfig} */
const nextConfig = {
  distDir: process.env.NEXT_DIST_DIR || ".next",
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000"
      },
      {
        protocol: "http",
        hostname: "127.0.0.1",
        port: "8000"
      }
    ]
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${internalApiBase}/api/:path*`
      }
    ];
  }
};

export default nextConfig;
