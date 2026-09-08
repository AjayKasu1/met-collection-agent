"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

import type { Citation, LiveObject } from "@/lib/types";
import { parseLiveObject, safeHttpUrl } from "@/lib/validation";

function sourceLabel(url: string): string {
  try {
    const parsed = new URL(url);
    if (
      parsed.hostname === "maps.metmuseum.org" &&
      parsed.pathname.startsWith("/navigate/")
    ) {
      return "The Met Interactive Map";
    }
    const segment = parsed.pathname.split("/").filter(Boolean).at(-1) ?? "Visitor information";
    return segment
      .replace(/[-_]/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  } catch {
    return "Visitor information";
  }
}

function ObjectCitation({ citation, index }: { citation: Citation; index: number }): React.ReactNode {
  const [object, setObject] = useState<LiveObject | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/objects/${citation.object_id}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) return null;
        return parseLiveObject(await response.json());
      })
      .then(setObject)
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) setObject(null);
      });
    return () => controller.abort();
  }, [citation.object_id]);

  const objectUrl = object?.source_url
    ? safeHttpUrl(object.source_url)
    : `https://www.metmuseum.org/art/collection/search/${citation.object_id}`;
  const imageUrl = object?.primary_image_small ? safeHttpUrl(object.primary_image_small) : null;

  return (
    <article className="citation-card object-card">
      <div className="citation-image" aria-hidden={!imageUrl}>
        {imageUrl ? (
          <Image alt="" fill sizes="(max-width: 680px) 72px, 88px" src={imageUrl} />
        ) : (
          <span>{String(index + 1).padStart(2, "0")}</span>
        )}
      </div>
      <div className="citation-copy">
        <p className="citation-kicker">Collection object {citation.object_id}</p>
        <h3>{object?.title ?? "Loading object details"}</h3>
        {object && (
          <p className="object-byline">
            {[object.artist, object.object_date].filter(Boolean).join(" · ") || "The Met Collection"}
          </p>
        )}
        {object?.is_on_view && (
          <p className="on-view">
            On view{object.gallery_number ? ` · Gallery ${object.gallery_number}` : ""}
          </p>
        )}
        <blockquote>{citation.quote}</blockquote>
        {objectUrl && (
          <a href={objectUrl} rel="noreferrer" target="_blank">
            View collection record <span aria-hidden="true">↗</span>
          </a>
        )}
      </div>
    </article>
  );
}

function VisitorCitation({ citation, index }: { citation: Citation; index: number }): React.ReactNode {
  const safeUrl = citation.source_url ? safeHttpUrl(citation.source_url) : null;
  return (
    <article className="citation-card visitor-card">
      <div className="source-number">{String(index + 1).padStart(2, "0")}</div>
      <div className="citation-copy">
        <p className="citation-kicker">Visitor information</p>
        <h3>{citation.source_url ? sourceLabel(citation.source_url) : "The Met"}</h3>
        <blockquote>{citation.quote}</blockquote>
        {safeUrl && (
          <a href={safeUrl} rel="noreferrer" target="_blank">
            Read source <span aria-hidden="true">↗</span>
          </a>
        )}
      </div>
    </article>
  );
}

export function CitationStrip({ citations }: { citations: Citation[] }): React.ReactNode {
  if (citations.length === 0) return null;
  return (
    <section aria-label="Sources" className="citation-section">
      <div className="section-label">
        <span>Sources</span>
        <span>{citations.length}</span>
      </div>
      <div className="citation-strip">
        {citations.map((citation, index) =>
          citation.object_id ? (
            <ObjectCitation citation={citation} index={index} key={`object-${citation.object_id}-${index}`} />
          ) : (
            <VisitorCitation citation={citation} index={index} key={`source-${citation.source_url}-${index}`} />
          ),
        )}
      </div>
    </section>
  );
}
