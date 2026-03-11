---
name: google-job-finder
description: Collect matching jobs from Google Jobs search results, save exactly target_job_count new matching jobs unless results run out, and report completion. Use when the user wants to find and save job listings from Google.
---

Use this skill when collecting Google Jobs results for a search query and job profile.
Use `target_job_count` from the mission or base knowledge when it is provided. If `target_job_count` is missing, default to `10`.

See [Google Jobs layout](references/google-jobs-layout.md) for how the page is structured.
See [troubleshooting](references/troubleshooting.md) if you encounter captchas, redirects, or empty results.

## Event Emission
- Emit events only for factual, already-completed milestones.
- Because Agent Events dispatch only after a successful tool action, attach each event to the successful action that confirms the milestone.
- Use these events:
  - `jobs_page_ready`
  - `job_saved`
  - `job_rejected`
  - `job_collection_done`
  - `job_collection_failed`
  - `job_collection_error`
- Emit `jobs_page_ready` once you have confirmed that Google Jobs is open and the `Jobs` tab is highlighted.
  - If no click was needed because the tab was already highlighted, attach this event to the next successful action while still on the Google Jobs page.
  - Payload:
    - `jobs_tab_highlighted: true`
- After `process_focused_job` returns `saved=true`, the next successful action must emit `job_saved`.
  - Prefer attaching it to `think(next_action="advance")`.
  - Payload should include:
    - `job_title`
    - `file_name`
- After `process_focused_job` returns a non-saved outcome, the next successful action must emit `job_rejected`.
  - Attach it to the next successful action taken after reading the result.
  - Use `reason_code` values such as:
    - `duplicate`
    - `profile_mismatch`
    - `unprocessable_missing_required_fields`
- Emit `job_collection_done` only for terminal non-failure outcomes.
  - Use `outcome` values `complete`, `partial`, or `empty`.
- Emit `job_collection_failed` only for terminal unrecoverable failures.
- Emit `job_collection_error` for recoverable workflow problems such as redirects, temporary blockers, or unexpected page states.

## Entry
1. Call `open_url` with exactly `https://www.google.com/search?q={search query}&udm=8`.
   - Every Google search URL in this skill must include `&udm=8`.
   - Never call `open_url` with a Google search URL that omits `udm=8`.
2. If a captcha is visible, call `ask_user` with one option only: `["Done"]`.
   - If `ask_user` fails, call `flag`, emit:
     - `emit_events=[{"name":"job_collection_failed","data":{"failure_code":"captcha_unresolved","reason":"captcha could not be cleared by the user"}}]`
     then call `think(next_action="done")`.
3. If the `Jobs` tab is already selected or highlighted, do not click it again; continue to the next step.
   If the `Jobs` tab is visible and not already selected, click it.
   If no `Jobs` tab is visible, call `flag`, emit:
   - `emit_events=[{"name":"job_collection_failed","data":{"failure_code":"jobs_tab_missing","reason":"Google search results did not expose a Jobs tab"}}]`
   then call `think(next_action="done")`.
4. Once the Google Jobs page is open and the `Jobs` tab is highlighted, emit:
   - `emit_events=[{"name":"jobs_page_ready","data":{"jobs_tab_highlighted":true}}]`
5. If the Google Jobs page is open but no job listings are visible in the left panel, call `report_data` saying there were no job listings for the requested search query and job profile, emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"empty","saved_count":0,"target_count":target_job_count}}]`
   then call `think(next_action="done")`.

## Collection Loop
1. Call `think(next_action="start_loop", loop_count=target_job_count, loop_description="Save target_job_count new matching Google jobs")`.
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
   - `process_focused_job` saves `job_summary` as a concise summary of the most important visible parts of the role, not the full raw job description.
4. React to the result:
   - `saved=true` -> call `think(next_action="advance")` and emit:
     - `emit_events=[{"name":"job_saved","data":{"job_title":"<job_title>","file_name":"google-jobs-list.jsonl"}}]`
   - `duplicate=true` -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"duplicate","reason":"job already exists in the saved jobs file"}}]`
   - `matches_profile=false` -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"profile_mismatch","reason":"focused job did not match the requested profile"}}]`
   - `processable=false` -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"unprocessable_missing_required_fields","reason":"focused job was missing required fields"}}]`
   - After emitting `job_rejected`, do not advance; continue to the next unseen card
5. Repeat from step 2 until `target_job_count` jobs are saved or results are exhausted.

## Recovery
If at any point you leave the Google Jobs page:
1. On the successful recovery or notification action for a recoverable problem, emit `job_collection_error`.
   - Example payload:
     - `emit_events=[{"name":"job_collection_error","data":{"error_code":"redirected_off_jobs_page","message":"Left the Google Jobs page and started recovery","recoverable":true}}]`
2. Call `go_back`.
3. If that doesn't return to Google Jobs, call `open_url` with the original search URL (with `&udm=8`).
4. If the Jobs tab needs re-selecting, click it.
5. If the page shows no listings after recovery, call `report_data`, emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"partial","saved_count":<saved_count>,"target_count":target_job_count}}]`
   and end.

Do not call `think(next_action="stuck")` for duplicates, skipped jobs, or lack of new cards.
Instead: scroll the left panel for more cards, or recover navigation if off-page.

## Completion
1. If results run out before `target_job_count` jobs, call `report_data` with partial completion details and emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"partial","saved_count":<saved_count>,"target_count":target_job_count}}]`
2. If `target_job_count` jobs are saved, call `report_data` with completion details and emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"complete","saved_count":<saved_count>,"target_count":target_job_count}}]`
3. If the workflow cannot continue because of an unrecoverable blocker, call `flag`, emit:
   - `emit_events=[{"name":"job_collection_failed","data":{"failure_code":"<code>","reason":"<why the mission cannot continue>"}}]`
   then call `think(next_action="done")`.
4. If a recipient email is available, call `send_email` with the same summary.
5. Call `think(next_action="done")`.
