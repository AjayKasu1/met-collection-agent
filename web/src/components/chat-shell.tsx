"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { CitationStrip } from "@/components/citation-strip";
import { CollectionMark } from "@/components/mark";
import { ProvenancePanel } from "@/components/provenance-panel";
import { messageProvenance, messageText } from "@/lib/messages";
import type { Language, MuseumMessage } from "@/lib/types";

const examples = [
  "Show me a Van Gogh landscape in the collection",
  "What can I see in Gallery 131?",
  "Is the Temple of Dendur on view today?",
  "What should families know before visiting?",
];

const languages: { value: Language; label: string; hint: string }[] = [
  { value: "en", label: "EN", hint: "English" },
  { value: "fr", label: "FR", hint: "Français" },
  { value: "es", label: "ES", hint: "Español" },
  { value: "zh", label: "中文", hint: "中文" },
];

function newSessionId(): string {
  return crypto.randomUUID();
}

export function ChatShell(): React.ReactNode {
  const transport = useMemo(() => new DefaultChatTransport<MuseumMessage>({ api: "/api/chat" }), []);
  const { messages, sendMessage, status, error, stop, setMessages, clearError } = useChat<MuseumMessage>({
    transport,
    throttle: 40,
  });
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState<Language>("en");
  const [sessionId, setSessionId] = useState(newSessionId);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = status === "submitted" || status === "streaming";

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, status]);

  async function submit(question: string): Promise<void> {
    const text = question.trim();
    if (!text || busy) return;
    clearError();
    setInput("");
    await sendMessage({ text }, { body: { session_id: sessionId, language } });
  }

  function onSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void submit(input);
  }

  function reset(): void {
    if (busy) void stop();
    setMessages([]);
    clearError();
    setInput("");
    setSessionId(newSessionId());
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Open Collection home">
          <CollectionMark />
          <span>
            <strong>Open Collection</strong>
            <small>A research guide to The Met</small>
          </span>
        </a>
        <div className="header-actions">
          <div aria-label="Answer language hint" className="language-selector" role="group">
            {languages.map((option) => (
              <button
                aria-label={option.hint}
                aria-pressed={language === option.value}
                key={option.value}
                onClick={() => setLanguage(option.value)}
                type="button"
              >
                {option.label}
              </button>
            ))}
          </div>
          <button className="new-chat" onClick={reset} type="button">
            New conversation
          </button>
        </div>
      </header>

      <div className="conversation" id="top">
        {messages.length === 0 ? (
          <section className="welcome">
            <p className="eyebrow">Explore 20,000 collection records</p>
            <h1>Ask the collection.<br />Follow the evidence.</h1>
            <p className="welcome-copy">
              Find artworks, compare objects, and plan a visit. Every factual answer is checked against collection records or current visitor information.
            </p>
            <div className="example-grid" aria-label="Example questions">
              {examples.map((example, index) => (
                <button disabled={busy} key={example} onClick={() => void submit(example)} type="button">
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {example}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section aria-label="Conversation" className="messages">
            {messages.map((message) => {
              const provenance = messageProvenance(message);
              return (
                <article className={`message message-${message.role}`} key={message.id}>
                  <div className="message-label">{message.role === "user" ? "You" : "Collection guide"}</div>
                  <div className="message-body">
                    <p className="answer-text">{messageText(message)}</p>
                    {provenance?.answer.handoff && (
                      <aside className="handoff">
                        <strong>Staff help recommended</strong>
                        <span>{provenance.answer.handoff.reason}</span>
                        <a href={`mailto:${provenance.answer.handoff.suggested_contact}`}>
                          {provenance.answer.handoff.suggested_contact}
                        </a>
                      </aside>
                    )}
                    {provenance && <CitationStrip citations={provenance.answer.citations} />}
                    {provenance && <ProvenancePanel provenance={provenance} />}
                  </div>
                </article>
              );
            })}
            {busy && (
              <div aria-live="polite" className="working-status" role="status">
                <span className="status-pulse" />
                {status === "submitted" ? "Reviewing the collection" : "Preparing a verified answer"}
              </div>
            )}
            {error && (
              <div aria-live="assertive" className="error-notice" role="alert">
                <span>{error.message}</span>
                <button onClick={clearError} type="button">Dismiss</button>
              </div>
            )}
            <div ref={endRef} />
          </section>
        )}
      </div>

      <div className="composer-wrap">
        <form className="composer" onSubmit={onSubmit}>
          <label className="sr-only" htmlFor="question">Ask about the collection or your visit</label>
          <textarea
            autoFocus
            id="question"
            maxLength={4000}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask about an artwork, artist, gallery, or visit…"
            rows={2}
            value={input}
          />
          {busy ? (
            <button aria-label="Stop generating" className="send-button stop-button" onClick={() => void stop()} type="button">
              <span aria-hidden="true" />
            </button>
          ) : (
            <button aria-label="Send question" className="send-button" disabled={!input.trim()} type="submit">
              <span aria-hidden="true">↑</span>
            </button>
          )}
        </form>
        <p className="composer-note">Language is detected automatically. Verify gallery locations with museum staff.</p>
      </div>

      <footer>
        <span>Independent collection guide</span>
        <span>Not affiliated with The Met · Collection data CC0</span>
      </footer>
    </main>
  );
}
