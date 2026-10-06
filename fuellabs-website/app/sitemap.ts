import type { MetadataRoute } from "next";

const siteUrl = "https://fuellabs.in";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date("2026-10-05");

  return [
    { url: siteUrl, lastModified, changeFrequency: "weekly", priority: 1 },
    { url: `${siteUrl}/hobbylm`, lastModified, changeFrequency: "weekly", priority: 0.8 },
    { url: `${siteUrl}/research`, lastModified, changeFrequency: "weekly", priority: 0.8 },
    { url: `${siteUrl}/research/pretraining-hobbylm`, lastModified, changeFrequency: "monthly", priority: 0.9 },
    { url: `${siteUrl}/team`, lastModified, changeFrequency: "monthly", priority: 0.6 },
  ];
}
