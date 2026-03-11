import type { Metadata } from "next";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk"
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-ibm-plex-mono",
  weight: ["400", "500"]
});

export const metadata: Metadata = {
  title: "记忆工程测试工作台",
  description: "上传最多 100 张照片，查看人脸识别结果、任务状态与失败图片记录。"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body className={`${spaceGrotesk.variable} ${ibmPlexMono.variable}`}>{children}</body>
    </html>
  );
}
