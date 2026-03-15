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
  - `job_results_exhausted`
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
    - `dedupe_key`
  - Always copy `dedupe_key` exactly from the successful `process_focused_job` result. Do not invent or recompute it yourself.
  - This still applies when the saved job reaches `target_job_count`.
  - After that `job_saved` has been emitted, do not emit `job_saved` again unless a later `process_focused_job` returns `saved=true` for a new job.
  - Never emit `job_collection_done` on the same successful action that must emit `job_saved` for the most recent save.
  - `job_collection_done(outcome="complete")` may only be emitted on a later successful action after that `job_saved` event has already been emitted.
- After `process_focused_job` returns a non-saved outcome, the next successful action must emit `job_rejected`, except for recoverable workflow problems such as `search_context_mismatch`, which must emit `job_collection_error` instead.
  - Attach it to the next successful action taken after reading the result.
  - Use `reason_code` values such as:
    - `duplicate`
    - `profile_mismatch`
    - `unprocessable_missing_required_fields`
- Emit `job_collection_done` only for terminal non-failure outcomes.
  - Use `outcome` values `complete`, `partial`, or `empty`.
- Emit `job_results_exhausted` only after you have explicitly checked for more unseen cards, attempted recovery or scrolling if needed, and confirmed there are no more viable unseen results to inspect.
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
   - `emit_events=[{"name":"job_results_exhausted","data":{"confirmed":true}}]`
6. On the next successful action, call `report_data` with the empty-results summary and emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"empty","saved_count":0,"target_count":target_job_count}}]`

## Collection Loop
1. Start the collection loop before inspecting the first card:
   - `think(next_action="start_loop", loop_mode="until_done", loop_description="Inspect distinct Google Jobs cards and save matching jobs", loop_exit_condition="Stop only when mission progress allows complete completion, or when results are explicitly exhausted and mission progress allows partial or empty completion")`
   - This loop organizes repeated job inspection from round 1 onward.
   - Mission completion is controlled by mission progress and terminal events, not by loop rounds.
2. For each loop round, click the next unseen job card in the left panel.
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
   - Before calling `process_focused_job`, inspect the visible job card and details panel.
   - If the job visibly states that access is restricted to premium members, members-only users, subscribers, special-access users, or any similar gated audience, do not save it.
   - Treat this as a semantic access restriction rule, not a fixed badge denylist.
   - Examples that indicate this kind of restriction include text like `For Premium Members only`, `Members only`, or similar gated-access wording.
4. Inside the active until-done loop, react to the result:
   - `saved=true` and mission progress does not yet allow `job_collection_done(outcome="complete")` -> call `think(next_action="advance")` and emit:
     - `emit_events=[{"name":"job_saved","data":{"job_title":"<job_title>","file_name":"google-jobs-list.jsonl","dedupe_key":"<returned dedupe_key>"}}]`
   - `saved=true` and this save now satisfies `target_job_count` -> do not skip straight to terminal completion.
     - On the next successful action, call `think(next_action="end_loop")` and emit:
       - `emit_events=[{"name":"job_saved","data":{"job_title":"<job_title>","file_name":"google-jobs-list.jsonl","dedupe_key":"<returned dedupe_key>"}}]`
     - On a later successful action after that, proceed to terminal `job_collection_done(outcome="complete")`.
   - `reason="search_context_mismatch"` or `search_context_valid=false` -> this is a recoverable workflow problem, not a rejected job.
     - On the next successful recovery or notification action emit:
       - `emit_events=[{"name":"job_collection_error","data":{"error_code":"search_context_mismatch","message":"Google Jobs search drifted away from the original query; starting recovery","recoverable":true}}]`
     - Use recent navigation history first:
       - If the previous page or most recent back target appears to be the original Google Jobs results page for the requested search query, call `go_back`.
       - Re-check the current page after `go_back`.
       - If `go_back` does not restore the original Google Jobs search context, call `open_url` with the original search URL `https://www.google.com/search?q={search query}&udm=8`.
     - After recovery, confirm the Google Jobs page is back on the original search context before continuing collection.
   - `restricted_access=true` or visible premium/member-only restriction -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"restricted_access","reason":"job requires premium, member-only, subscriber, or similar restricted access"}}]`
   - `duplicate=true` -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"duplicate","reason":"job already exists in the saved jobs file"}}]`
   - `matches_profile=false` -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"profile_mismatch","reason":"focused job did not match the requested profile"}}]`
   - `processable=false` -> on the next successful action emit:
     - `emit_events=[{"name":"job_rejected","data":{"reason_code":"unprocessable_missing_required_fields","reason":"focused job was missing required fields"}}]`
   - After emitting `job_rejected`, do not advance immediately unless you completed a meaningful user-facing round and are continuing the until-done loop.
   - In `loop_mode="until_done"`, call `think(next_action="advance")` only after a meaningful user-facing round such as opening a new distinct card, processing it, or scrolling to fetch more unseen cards.
5. If there are no more visible unseen cards, scroll the left panel and check again before considering exhaustion.
   - If you have explicitly checked for more unseen cards, attempted recovery or scrolling if needed, and confirmed there are no more viable unseen results to inspect, emit:
     - `emit_events=[{"name":"job_results_exhausted","data":{"confirmed":true}}]`
6. Do not emit `job_collection_done` just because you saw duplicates, rejected jobs, or a short run of ineligible cards.
   - Duplicate and ineligible jobs do not count toward the target.
   - Continue to another unseen card unless results have been explicitly exhausted.
7. Stay inside the until-done loop until one of these is true:
   - mission progress allows `job_collection_done(outcome="complete")`
   - `job_results_exhausted` has been emitted and mission progress allows `job_collection_done(outcome="partial")` or `job_collection_done(outcome="empty")`
   - an unrecoverable failure requires `job_collection_failed`

## Recovery
If at any point you leave the Google Jobs page, or the page stays on Google Jobs but drifts to the wrong search context:
1. On the successful recovery or notification action for a recoverable problem, emit `job_collection_error`.
   - Example payload:
     - `emit_events=[{"name":"job_collection_error","data":{"error_code":"redirected_off_jobs_page","message":"Left the Google Jobs page and started recovery","recoverable":true}}]`
2. Check recent navigation history first.
   - If the previous page or most recent back target looks like the original Google Jobs results page for the requested search, call `go_back`.
3. If `go_back` does not return to the original Google Jobs search context, call `open_url` with the original search URL (with `&udm=8`).
4. If the Jobs tab needs re-selecting, click it.
5. If the page shows no listings after recovery, on the next successful action emit:
   - `emit_events=[{"name":"job_results_exhausted","data":{"confirmed":true}}]`
6. Only after `job_results_exhausted` has already been emitted on a prior successful action may you emit a terminal `job_collection_done` partial or empty outcome.

Do not call `think(next_action="stuck")` for duplicates, skipped jobs, or lack of new cards.
Instead: scroll the left panel for more cards, or recover navigation if off-page.

## Completion
1. If results run out before `target_job_count` jobs, first use a successful action to emit:
   - `emit_events=[{"name":"job_results_exhausted","data":{"confirmed":true}}]`
2. On the next successful action after `job_results_exhausted`, call `report_data` with partial completion details and emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"partial","saved_count":<saved_count>,"target_count":target_job_count}}]`
3. If `target_job_count` jobs are saved, call `report_data` with completion details and emit:
   - `emit_events=[{"name":"job_collection_done","data":{"outcome":"complete","saved_count":<saved_count>,"target_count":target_job_count}}]`
   - If the last saved job was processed on the immediately prior step, make sure that prior successful action emitted `job_saved` first.
   - `job_collection_done(outcome="complete")` must happen on a later successful action, never on the same successful action that emitted the final required `job_saved`.
4. If the workflow cannot continue because of an unrecoverable blocker, call `flag`, emit:
   - `emit_events=[{"name":"job_collection_failed","data":{"failure_code":"<code>","reason":"<why the mission cannot continue>"}}]`
   then call `think(next_action="done")`.
5. If a recipient email is available, call `send_email` with the same summary.
6. Do not emit `job_collection_done(outcome="partial")` or `job_collection_done(outcome="empty")` on the same action as `job_results_exhausted`.
   - First emit `job_results_exhausted`.
   - Then use a later successful action for the terminal `job_collection_done` event.
7. Do not call `think(next_action="done")` merely to exit the loop.
   - `think(next_action="done")` is only for mission completion after terminal success or failure conditions are satisfied.
   - Use mission events and mission progress to decide whether completion is allowed.
8. After terminal `job_collection_done` or `job_collection_failed`, stop.
