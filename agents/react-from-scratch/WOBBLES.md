# Where local tool-calling wobbles

Same agent (`run_live.decide`, tools `search` + `calc`, `max_steps=6`), same goals,
two models. Runs on 2026-11-14: `gpt-5.5` (OpenAI) vs `qwen3.5:9b` (Ollama, 16 GB Mac).

| # | Goal | `gpt-5.5` | `qwen3.5:9b` |
| --- | --- | --- | --- |
| 0 | 15% of Tokyo's population | search → calc, 5,567,250 | same steps, same answer |
| 1 | Canada vs California, by how many? | 2 searches → calc: **377,656** | 2 searches → calc: **614,244** |
| 2 | (17^3 - 2^10) / 7, 2 decimals | calc `**`, 555.57 | same |
| 3 | Capital of France | no tool, Paris | same |
| 4 | Eiffel Towers in Everest | **no search**, calc from memory, ~26.8 | 2 searches → calc, ~26.8 |
| 5 | Apple's price next Friday | no tool, "can't know" | searched, "can't know" + lists analyst targets |

## What wobbled

- **Not the tool-call format.** 0 unknown tools, 0 bad args, 0 JSON errors on
  either model. Both turned `^` into `**` for the calculator by themselves.
- **Facts from the same search results.** Goal 1: both got the same results but
  read a different California number (39,364,774 vs 39,128,186), so the answers
  differ by ~236k. Neither said which year or source it used. Goal 0: both used
  Greater Tokyo (37.1 M), not Tokyo city (~14 M), without saying so.
- **Frontier skipped the tool.** Goal 4: `gpt-5.5` answered from memory instead
  of searching. Right numbers this time, but nothing in the history backs them.
- **Local repeats web text.** Goal 5: `qwen3.5:9b` searched and passed on
  analyst price targets from untrusted pages; the "untrusted" label didn't stop
  it quoting them (it still refused to predict).
- **Speed.** Local took 5–65 s per goal; `gpt-5.5` 1–13 s. About 5× slower.

## For validate-and-retry (task 7)

No bad tool calls showed up, so nothing here to retry yet. The known failure
kinds to guard anyway: unknown tool name, missing/extra args, args of the wrong
type. Test them with a scripted `decide`, not by waiting for the model to fail.
