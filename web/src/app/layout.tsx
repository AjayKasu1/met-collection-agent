import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Open Collection",
  description: "A cited research guide to The Metropolitan Museum of Art collection.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>): React.ReactNode {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
