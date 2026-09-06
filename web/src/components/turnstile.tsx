"use client";

import Script from "next/script";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";

type WidgetId = string;
type TurnstileApi = {
  render: (
    container: HTMLElement,
    options: {
      sitekey: string;
      action: string;
      callback: (token: string) => void;
      "error-callback": () => void;
      "expired-callback": () => void;
      theme: "auto";
    },
  ) => WidgetId;
  remove: (widgetId: WidgetId) => void;
  reset: (widgetId: WidgetId) => void;
};

declare global {
  interface Window {
    turnstile?: TurnstileApi;
  }
}

export type TurnstileHandle = { reset: () => void };

type Props = {
  action: string;
  onToken: (token: string) => void;
  siteKey: string;
};

export const Turnstile = forwardRef<TurnstileHandle, Props>(function Turnstile(
  { action, onToken, siteKey },
  ref,
) {
  const container = useRef<HTMLDivElement>(null);
  const widgetId = useRef<WidgetId | null>(null);

  const render = useCallback((): void => {
    if (!window.turnstile || !container.current || widgetId.current !== null) return;
    widgetId.current = window.turnstile.render(container.current, {
      sitekey: siteKey,
      action,
      callback: onToken,
      "error-callback": () => onToken(""),
      "expired-callback": () => onToken(""),
      theme: "auto",
    });
  }, [action, onToken, siteKey]);

  useImperativeHandle(ref, () => ({
    reset: () => {
      if (widgetId.current && window.turnstile) window.turnstile.reset(widgetId.current);
      onToken("");
    },
  }));

  useEffect(() => {
    render();
    return () => {
      if (widgetId.current && window.turnstile) window.turnstile.remove(widgetId.current);
      widgetId.current = null;
    };
  }, [render]);

  return (
    <>
      <Script
        onReady={render}
        src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"
        strategy="afterInteractive"
      />
      <div aria-label="Security verification" className="turnstile" ref={container} />
    </>
  );
});
