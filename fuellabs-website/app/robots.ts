import type { MetadataRoute } from "next";

const siteUrl = "https://fuellabs.in";
const allowIndexing = process.env.VERCEL_ENV === "production";

export default function robots(): MetadataRoute.Robots {
  if (!allowIndexing) return { rules: { userAgent: "*", disallow: "/" } };

  return {
    rules: [
      { userAgent: "*", allow: "/" },
      { userAgent: "*", disallow: "/style-guide" },
    ],
    sitemap: `${siteUrl}/sitemap.xml`,
  };
}
