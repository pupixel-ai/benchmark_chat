import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope, Newsreader } from "next/font/google";
import "./globals.css";

const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-newsreader"
});

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope"
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
      <body className={`${newsreader.variable} ${manrope.variable} ${ibmPlexMono.variable}`}>{children}</body>
    </html>
  );
}
