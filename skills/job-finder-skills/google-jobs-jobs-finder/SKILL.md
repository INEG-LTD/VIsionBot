---
name: google-job-finder
description: Collect matching jobs from Google Jobs search results, save exactly 10 new matching jobs unless results run out, and report completion. Use when the user wants to find and save job listings from Google.
---

Use this skill when collecting Google Jobs results for a search query and job profile.

See [Google Jobs layout](references/google-jobs-layout.md) for how the page is structured.
See [troubleshooting](references/troubleshooting.md) if you encounter captchas, redirects, or empty results.

## Entry
1. Call `open_url` with exactly `https://www.google.com/search?q={search query}&udm=8`.
   - Every Google search URL in this skill must include `&udm=8`.
   - Never call `open_url` with a Google search URL that omits `udm=8`.
2. If a captcha is visible, call `ask_user` with one option only: `["Done"]`.
   - If `ask_user` fails, call `flag` and then `think(next_action="done")`.
3. If the `Jobs` tab is already selected or highlighted, do not click it again; continue to the next step.
   If the `Jobs` tab is visible and not already selected, click it.
   If no `Jobs` tab is visible, call `flag` and then `think(next_action="done")`.
4. If the Google Jobs page is open but no job listings are visible in the left panel, call `report_data` saying there were no job listings for the requested search query and job profile, then call `think(next_action="done")`.

## Collection Loop
1. Call `think(next_action="start_loop", loop_count=10, loop_description="Save 10 new matching Google jobs")`.
2. Click the next unseen job card in the left panel.
   - If no unseen card is visible, scroll the left panel with `scroll_container`.
   - Only use `scroll_down` if `scroll_container` is not possible.
   - If the clicked card was the last visible card, scroll the left panel before the next selection.
   - After any scroll, re-read visible cards before continuing. Do not rely on pre-scroll state.
   - If two consecutive cards from the current visible set are duplicates or unprocessable, scroll before trying another card from that area.
3. Once the job details load on the right panel, call `process_focused_job` with:
   - `job_profile`
   - `file_name="google-jobs-list.jsonl"`
   - `search_query`
4. React to the result:
   - `saved=true` -> call `think(next_action="advance")`
   - `duplicate=true`, `matches_profile=false`, or `processable=false` -> do not advance, go to next unseen card
5. Repeat from step 2 until 10 jobs are saved or results are exhausted.

## Recovery
If at any point you leave the Google Jobs page:
1. Call `go_back`.
2. If that doesn't return to Google Jobs, call `open_url` with the original search URL (with `&udm=8`).
3. If the Jobs tab needs re-selecting, click it.
4. If the page shows no listings after recovery, report partial results and end.

Do not call `think(next_action="stuck")` for duplicates, skipped jobs, or lack of new cards.
Instead: scroll the left panel for more cards, or recover navigation if off-page.

## Completion
1. If results run out before 10 jobs, report partial completion with the count saved.
2. Call `report_data` with: search query, number of new jobs saved, file name `google-jobs-list.jsonl`.
3. If a recipient email is available, call `send_email` with the same summary.
4. Call `think(next_action="done")`.
