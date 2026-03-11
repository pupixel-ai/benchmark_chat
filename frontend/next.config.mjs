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
  }
};

export default nextConfig;
