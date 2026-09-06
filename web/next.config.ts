import type { NextConfig } from "next";

function validateProductionBuild(): void {
  if (process.env.NODE_ENV !== "production") return;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY?.trim();
  if (!apiUrl) throw new Error("NEXT_PUBLIC_API_URL is required for a production build");
  if (!turnstileSiteKey) {
    throw new Error("NEXT_PUBLIC_TURNSTILE_SITE_KEY is required for a production build");
  }
  let parsed: URL;
  try {
    parsed = new URL(apiUrl);
  } catch {
    throw new Error("NEXT_PUBLIC_API_URL must be a valid URL");
  }
  if (
    parsed.protocol !== "https:" ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash ||
    parsed.pathname !== "/"
  ) {
    throw new Error("NEXT_PUBLIC_API_URL must be an HTTPS origin without credentials or a path");
  }
}

validateProductionBuild();

const contentSecurityPolicy = [
  "default-src 'self'",
  "base-uri 'self'",
  "connect-src 'self' https://challenges.cloudflare.com",
  "font-src 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
  "frame-src https://challenges.cloudflare.com",
  "img-src 'self' data: https://images.metmuseum.org https://collectionapi.metmuseum.org",
  "object-src 'none'",
  "script-src 'self' 'unsafe-inline' https://challenges.cloudflare.com",
  "style-src 'self' 'unsafe-inline'",
  "upgrade-insecure-requests",
].join("; ");

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Content-Security-Policy", value: contentSecurityPolicy },
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Strict-Transport-Security", value: "max-age=31536000" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          {
            key: "Permissions-Policy",
            value: "camera=(), geolocation=(), microphone=()",
          },
        ],
      },
    ];
  },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "images.metmuseum.org" },
      { protocol: "https", hostname: "collectionapi.metmuseum.org" },
    ],
  },
};

export default nextConfig;
