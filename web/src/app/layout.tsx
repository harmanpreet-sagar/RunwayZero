import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, IBM_Plex_Sans_Condensed } from "next/font/google";
import "./globals.css";

const ui = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-ui" });
const head = IBM_Plex_Sans_Condensed({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-head" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Runway Zero",
  description: "Clear it or hold it. Make runway-crossing calls on real Boston Logan traffic and see if you beat the AI.",
};

// Apply a saved light/dark choice before first paint (otherwise the page flashes the system theme).
const themeScript = `try{var t=localStorage.getItem("rz_theme");if(t)document.documentElement.setAttribute("data-theme",t)}catch(e){}`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${ui.variable} ${head.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
