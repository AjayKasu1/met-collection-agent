import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WelcomePanel, welcomeCards } from "@/components/welcome-panel";

afterEach(cleanup);

describe("WelcomePanel", () => {
  it("offers four museum tasks with specific prompts", () => {
    render(<WelcomePanel disabled={false} onSelect={() => undefined} />);

    expect(screen.getByRole("heading", { name: /what would you like to discover/i })).toBeVisible();
    expect(screen.getAllByRole("button")).toHaveLength(4);
    expect(welcomeCards.map((card) => card.title)).toEqual([
      "Explore the collection",
      "Browse a gallery",
      "Plan your visit",
      "Accessibility",
    ]);
  });

  it("submits the prompt associated with the selected card", async () => {
    const onSelect = vi.fn<(prompt: string) => void>();
    const user = userEvent.setup();
    render(<WelcomePanel disabled={false} onSelect={onSelect} />);

    await user.click(screen.getByRole("button", { name: /browse a gallery/i }));

    expect(onSelect).toHaveBeenCalledOnce();
    expect(onSelect).toHaveBeenCalledWith("What can I see in Gallery 131?");
  });

  it("prevents selection while security verification is pending", () => {
    render(<WelcomePanel disabled onSelect={() => undefined} />);

    for (const card of screen.getAllByRole("button")) {
      expect(card).toBeDisabled();
    }
  });
});
