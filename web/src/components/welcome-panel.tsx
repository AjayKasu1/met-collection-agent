"use client";

import type { ReactNode } from "react";

type WelcomeCard = {
  description: string;
  icon: "accessibility" | "collection" | "gallery" | "visit";
  prompt: string;
  title: string;
};

export const welcomeCards: readonly WelcomeCard[] = [
  {
    title: "Explore the collection",
    description: "Find an artist, artwork, period, or material",
    prompt: "Show me a Van Gogh landscape in the collection",
    icon: "collection",
  },
  {
    title: "Browse a gallery",
    description: "See which collection objects are listed in a gallery",
    prompt: "What can I see in Gallery 131?",
    icon: "gallery",
  },
  {
    title: "Plan your visit",
    description: "Check policies, hours, admission, and family guidance",
    prompt: "What should families know before visiting?",
    icon: "visit",
  },
  {
    title: "Accessibility",
    description: "Find entrances, services, and access information",
    prompt: "Which entrance is wheelchair accessible?",
    icon: "accessibility",
  },
] as const;

type WelcomePanelProps = {
  disabled: boolean;
  onSelect: (prompt: string) => void;
};

function CardIcon({ icon }: Pick<WelcomeCard, "icon">): ReactNode {
  if (icon === "collection") {
    return (
      <svg aria-hidden="true" viewBox="0 0 32 32">
        <circle cx="14" cy="14" r="8.5" />
        <path d="m20.5 20.5 6 6" />
        <path d="M10.5 14h7M14 10.5v7" />
      </svg>
    );
  }
  if (icon === "gallery") {
    return (
      <svg aria-hidden="true" viewBox="0 0 32 32">
        <path d="M5 27V9l11-5 11 5v18" />
        <path d="M10 27V12h12v15M4 27h24" />
        <path d="M13 16h6M13 20h6" />
      </svg>
    );
  }
  if (icon === "visit") {
    return (
      <svg aria-hidden="true" viewBox="0 0 32 32">
        <rect height="21" rx="2" width="23" x="4.5" y="7" />
        <path d="M10 4v6M22 4v6M5 13h22" />
        <path d="m12 21 3 3 6-7" />
      </svg>
    );
  }
  return (
    <svg aria-hidden="true" viewBox="0 0 32 32">
      <circle cx="16" cy="6.5" r="3" />
      <path d="M7 12h18M16 10v8M11 28l5-10 5 10M8 17l8 1 8-1" />
    </svg>
  );
}

export function WelcomePanel({ disabled, onSelect }: WelcomePanelProps): ReactNode {
  return (
    <section aria-labelledby="welcome-title" className="welcome">
      <div className="welcome-intro">
        <p className="eyebrow">Independent collection guide</p>
        <h1 id="welcome-title">
          Welcome.<br />What would you like to discover?
        </h1>
        <p className="welcome-copy">
          Search 20,000 collection records and plan a visit with answers checked against museum sources.
        </p>
      </div>

      <div aria-label="Choose a starting point" className="welcome-card-grid">
        {welcomeCards.map((card) => (
          <button
            aria-label={`${card.title}: ${card.description}`}
            className="welcome-card"
            disabled={disabled}
            key={card.title}
            onClick={() => onSelect(card.prompt)}
            type="button"
          >
            <span className="welcome-card-icon"><CardIcon icon={card.icon} /></span>
            <span className="welcome-card-copy">
              <strong>{card.title}</strong>
              <span>{card.description}</span>
            </span>
            <span aria-hidden="true" className="welcome-card-arrow">&#8599;</span>
          </button>
        ))}
      </div>
    </section>
  );
}
