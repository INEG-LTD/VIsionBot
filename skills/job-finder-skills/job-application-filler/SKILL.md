---
name: job-application-filler
description: Fill and submit a single job application from a job or ATS URL using candidate facts, staged resume and cover letter files, and saved application preferences. Use when Codex needs to open an application page, detect Apply flows, complete multi-step ATS forms, upload documents, handle login or registration gates via user handoff, answer common application questions, and pause before the final submit click.
---

Complete exactly one job application per mission.
Use the application URL, staged file paths, candidate facts, and preferences from CUSTOM RULES as the source of truth.
Prefer the staged resume and cover letter files over the source CV path.
Use the resume fallback path from CUSTOM RULES only when the primary staged resume is rejected or the portal explicitly requires a different file type.

Open references only when needed:
- ATS-specific quirks: `references/ats-patterns.md`
- Question handling: `references/question-answering-guide.md`
- Consent handling: `references/consent-policy-guide.md`
- Manual follow-up cases: `references/manual-followup-guide.md`

## Core Rules

- First detect the current state before interacting.
- If the browser is not already on the target application flow, call `open_url` with the application URL from CUSTOM RULES.
- Treat redirects after `open_url`, Apply clicks, Continue clicks, or email-entry steps as expected until the destination page stabilizes.
- Re-detect the page after every redirect and classify the new page before continuing.
- If you are on a job posting page instead of the form, find and click the primary Apply button.
- Use `select_option` for dropdowns and custom selects.
- Use `upload_file` for resume and cover letter uploads.
- Use `read_file` for the staged cover letter text file when a text area requires the cover letter body.
- Never use site-provided resume builders, cover-letter builders, profile import, or AI-generated application-document services.
- Always prefer continuing with the user's staged resume and staged cover letter files from CUSTOM RULES.
- If the site does not allow continuing with the user's own resume or cover letter and only offers builder/import services, fail the application.
- If the job page, redirect destination, or application flow visibly states that the role is restricted to premium members, members-only users, subscribers, special-access users, or any similar gated audience, fail the application immediately.
- Treat Cloudflare, "Checking your browser", rate-limit pages, and similar anti-bot interstitials as auth handoff blockers.
- Skip unknown optional fields.
- Ask the user about unknown required fields.
- Never invent legal status, technical experience, credentials, dates, compensation facts, or anything not grounded in CUSTOM RULES or visible page content.
- Never call `think(next_action="done")` until one terminal application event has already been accepted.
- Do not click the final submit button more than once unless the page clearly returns to an editable validation-error state.
- Always ask the user before the final submit click.

## Event Emission

- Emit only factual, completed milestones.
- Use these events:
  - `application_form_detected`
  - `application_auth_required`
  - `application_page_filled`
  - `application_file_uploaded`
  - `application_review_reached`
  - `application_submit_attempted`
  - `application_submitted`
  - `application_cancelled`
  - `application_manual_followup_required`
  - `application_failed`
  - `application_error`
- Emit `application_form_detected` once the first real application form is visible.
  - Example:
    `emit_events=[{"name":"application_form_detected","data":{"form_url":"https://jobs.example.com/apply","ats_type":"workday"}}]`
- Emit `application_auth_required` when login, registration, MFA, email verification, magic link, captcha, Cloudflare interstitial, or similar anti-bot gating requires the user.
  - Always include `data.auth_type` and `data.message`.
  - Example:
    `emit_events=[{"name":"application_auth_required","data":{"auth_type":"captcha","message":"captcha must be completed before the application can continue"}}]`
- Emit `application_page_filled` after a page or major step is successfully completed and you advance or confirm it.
  - Always include `data.page_number` and `data.fields_filled`.
  - If the UI does not show a step number, count completed major form pages starting at `1`.
  - Count only fields you actively filled or corrected on that step, not untouched prefills.
  - Example:
    `emit_events=[{"name":"application_page_filled","data":{"page_number":2,"fields_filled":6}}]`
- Emit `application_file_uploaded` after a resume or cover letter upload succeeds.
  - Always include `data.file_type` as `resume` or `cover_letter`.
  - Always include `data.file_name` as the uploaded basename, not the full path.
  - Example resume upload event:
    `emit_events=[{"name":"application_file_uploaded","data":{"file_type":"resume","file_name":"active-application-resume.pdf"}}]`
  - Example cover letter upload event:
    `emit_events=[{"name":"application_file_uploaded","data":{"file_type":"cover_letter","file_name":"active-application-cover-letter.pdf"}}]`
- Emit `application_review_reached` when the final review step or final submit screen is visible.
  - Example:
    `emit_events=[{"name":"application_review_reached","data":{"review_visible":true}}]`
- Emit `application_submit_attempted` immediately after clicking the final submit button.
  - Example:
    `emit_events=[{"name":"application_submit_attempted","data":{"submit_button_label":"Submit Application"}}]`
- Emit `application_submitted` only after a confirmation page or clear success message is visible.
  - Example:
    `emit_events=[{"name":"application_submitted","data":{"confirmation_detected":true,"confirmation_text":"Application submitted successfully"}}]`
- Emit `application_manual_followup_required` when the workflow stops at an assessment, scheduler, external verification, already-applied message, or another human-only next step.
  - Always include `data.followup_type` and `data.instructions`.
  - Example:
    `emit_events=[{"name":"application_manual_followup_required","data":{"followup_type":"assessment_required","instructions":"Complete the coding assessment from the next page before the application can proceed"}}]`
- Emit `application_failed` only for terminal unrecoverable failures.
  - Example:
    `emit_events=[{"name":"application_failed","data":{"failure_code":"builder_only_flow","reason":"the site only allows resume builder or import flows instead of direct document upload"}}]`
- Emit `application_error` for recoverable problems.
  - Always include `data.error_code`, `data.message`, and `data.recoverable`.
  - Example:
    `emit_events=[{"name":"application_error","data":{"error_code":"upload_retry_failed","message":"resume upload failed twice and needs user help","recoverable":true}}]`

## Entry

1. Ensure the target application flow is open.
2. Detect which of these states is visible:
   - job posting with Apply button
   - redirect or landing page that has not yet reached the real form
   - email-first gate that asks only for the candidate email before continuing
   - real application form
   - login or registration wall
   - captcha, Cloudflare, rate-limit, or another anti-bot gate
   - builder/import service that wants to create or rewrite the resume or cover letter
   - restricted-access job page that visibly requires premium/member/subscriber/special access
   - confirmation or already-applied state
   - unsupported dead end
3. If a redirect or landing page is visible, wait for it to settle, then re-detect the state.
4. If an email-first gate is visible and it only asks for the candidate email, fill the candidate email from CUSTOM RULES, continue, and re-detect the state.
5. If a login or registration wall is visible, go to `Auth Handoff`.
6. If a captcha or anti-bot gate is visible, go to `Auth Handoff`.
7. If a restricted-access job page is visible, fail immediately.
8. If a builder/import service is visible, go to `Resume Builder Rejection`.
9. If a job posting page is visible and Apply is available, click Apply and re-detect.
10. When the first real application form is visible, emit:
   - `emit_events=[{"name":"application_form_detected","data":{"form_url":"<current url>","ats_type":"<detected ATS or generic>"}}]`
11. Call `think(next_action="start_loop", loop_mode="until_done", loop_description="Fill one job application safely", loop_exit_condition="Stop only after one terminal application event has been accepted: application_submitted, application_cancelled, application_manual_followup_required, or application_failed")`.

## Decision Table

| Situation | Action |
| --- | --- |
| Login, registration, MFA, OTP, magic link, email verification, captcha, Cloudflare, or another anti-bot interstitial is required | Emit `application_auth_required`, call `ask_user`, then re-detect the page after the user returns |
| Job posting is visible instead of the form | Click Apply, then re-detect |
| A redirect lands on another employer, ATS, or application-host domain | Wait for the new page to stabilize, then re-detect and continue from the new state |
| The page asks only for the candidate email before the real form | Fill the candidate email from CUSTOM RULES, continue, then re-detect |
| Email entry leads to sign-in, account creation, OTP, magic link, or verification | Emit `application_auth_required`, call `ask_user`, then re-detect |
| The page visibly states that the role is premium-only, members-only, subscriber-only, special-access, or similarly restricted | Emit `application_failed`, call `flag`, then end |
| The site offers resume builder, cover-letter builder, profile import, autofill-from-service, or AI writing help | Do not use it; look for a path to continue with the user's staged files |
| The site only allows builder/import services and does not allow the user's own files | Emit `application_failed`, call `flag`, then end |
| Unknown required field | Call `ask_user` with a concise question and any visible options |
| Unknown optional field | Skip it |
| Resume upload field | Call `upload_file` with the staged resume path from CUSTOM RULES |
| Cover letter upload field | Call `upload_file` with the staged cover letter path from CUSTOM RULES |
| The primary resume upload is rejected because of file type or the portal explicitly requires another format | Use the resume fallback order from `Attachment Rules` |
| Cover letter text area | Call `read_file` on the staged cover letter text path, then paste the content |
| A required supporting document such as portfolio, transcript, certification, work sample, or visa document is requested | Use the staged supporting document path from CUSTOM RULES when available; otherwise ask once, then use manual follow-up if the document is still unavailable |
| Marketing, SMS, or talent pool consent | Apply the configured policy from CUSTOM RULES |
| Required terms, privacy acknowledgement, or accuracy certification | Accept it |
| EEO field with configured answer | Use the configured answer |
| EEO field without configured answer but with `Prefer not to say` visible | Select `Prefer not to say` |
| EEO field without either of the above | Call `ask_user` |
| Open-ended non-technical question and `free_text_mode=best_effort` | Draft a grounded answer from the candidate summary, visible job context, and staged cover letter text |
| Technical screening, legal declaration, or unsupported essay question | Call `ask_user` |
| Parsed prefilled fields are visible | Review them according to `prefill_review_mode` |
| Final review page or submit button is visible | Emit `application_review_reached`, summarize state, then ask the user before clicking Submit |
| Confirmation is visible after submit | Emit `application_submitted`, report the confirmation, then end |
| Assessment, scheduler, already-applied page, or external step blocks completion | Emit `application_manual_followup_required`, tell the user exactly what to do next, then end |

## Auth Handoff

1. Emit `application_auth_required` with the blocker type and a short message.
2. Call `ask_user`.
3. Use short options that match the situation, for example:
   - `["I've signed in", "Skip this application"]`
   - `["Done", "Skip this application"]`
   - `["I've verified my email", "Skip this application"]`
4. If the user skips, emit:
   - `emit_events=[{"name":"application_cancelled","data":{"reason":"user skipped during auth handoff"}}]`
   then call `think(next_action="done")`.
5. After the user returns, re-detect the state.

## Restricted Access Rejection

1. If the visible job page, redirect destination, or apply destination says the role is restricted to premium members, members-only users, subscribers, special-access users, or a similar gated audience, do not continue.
2. Treat this as a semantic restriction rule, not a fixed badge denylist.
3. Examples of this kind of restriction include wording like `For Premium Members only`, `Members only`, `Subscriber only`, or other visible gated-access statements.
4. Emit:
   - `emit_events=[{"name":"application_failed","data":{"failure_code":"restricted_access_job","reason":"the role is visibly restricted to premium, member-only, subscriber, or similar gated access"}}]`
   then call `flag` and `think(next_action="done")`.

## Resume Builder Rejection

1. If the page offers to build, rewrite, import, or generate the resume or cover letter:
   - do not use those flows
   - do not paste the user's resume into a site builder
   - do not accept profile-import or AI-generated document services as a substitute for the staged files
2. Look for a truthful path that still allows:
   - resume upload from the staged resume path
   - cover letter upload from the staged cover letter path
   - cover letter text entry from the staged cover letter text file
3. If such a path exists, use it and continue.
4. If no such path exists and the site forces builder/import services, emit:
   - `emit_events=[{"name":"application_failed","data":{"failure_code":"builder_only_flow","reason":"the site only allows resume or cover letter builder/import services instead of the user's own files"}}]`
   then call `flag` and `think(next_action="done")`.

## Attachment Rules

- Resume uploads:
  - Try the staged primary resume upload path first.
  - If the portal rejects that file type or explicitly requires another format, use the resume fallback path from CUSTOM RULES.
  - If CUSTOM RULES also provide a staged `.doc` or `.docx` resume path, use it only after the earlier options are truthfully disallowed.
  - Never switch to a fallback unless the page clearly rejects or disallows the earlier option.
- Cover letter uploads:
  - Use the staged cover letter upload path.
  - Use the staged cover letter text path only for text boxes that clearly request the cover letter body.
- Supporting documents:
  - If a required portfolio, transcript, certification, work sample, visa document, or similar attachment is requested and CUSTOM RULES provide a staged path, upload it.
  - If the document is required and no staged path exists, call `ask_user` once.
  - If the document is still unavailable after that handoff, emit:
    - `emit_events=[{"name":"application_manual_followup_required","data":{"followup_type":"missing_required_document","instructions":"Upload the required supporting document manually, then resume or restart the application"}}]`
    then call `think(next_action="done")`.

## Filling Loop

For each visible step:

1. Inspect the visible required fields first, then optional fields.
2. Fill deterministic profile fields from CUSTOM RULES:
   - name
   - email
   - phone
   - address
   - city, state, postcode, country
   - LinkedIn, GitHub, website, portfolio URL
   - work authorization
   - sponsorship needed
   - notice period
   - earliest start date
   - relocation willingness
   - travel willingness
   - desired salary
3. If the visible step is an email-first gate, fill only the candidate email, continue, and re-detect before treating it as the full form.
4. If the page redirects to another site or ATS during the flow, wait for the destination to stabilize and re-detect before continuing.
5. If the page visibly shows a premium/member/subscriber/special-access restriction, use `Restricted Access Rejection`.
6. If the page tries to route the candidate into a resume or cover-letter builder/import flow, use `Resume Builder Rejection`.
7. Upload files with the exact staged paths from CUSTOM RULES and follow `Attachment Rules` for any required fallback.
8. Follow the question and consent policies below.
9. Fix any visible validation errors before advancing.
10. Advance with the visible `Next`, `Continue`, `Review`, or equivalent action.
11. Emit `application_page_filled` after the step succeeds.
12. Re-detect the next state.

## Question Rules

- Work authorization and sponsorship:
  - Use the stored answers from CUSTOM RULES.
- Candidate email:
  - Use the stored candidate email from CUSTOM RULES for normal application fields and email-first gates.
  - Do not treat an email-only gate as auth unless the page then requires sign-in, account creation, OTP, magic link, or verification.
- Salary:
  - Use the stored salary or matching range when it is present.
  - If the form forces a salary answer and none is provided in CUSTOM RULES, ask the user.
- Availability and relocation:
  - Use stored values when present.
  - Otherwise ask the user if the field is required.
- Travel:
  - Use stored values when present.
  - Otherwise ask the user if the field is required.
- Website, portfolio, LinkedIn, and GitHub:
  - Use the stored field that truthfully matches what the form is asking for.
  - Do not substitute LinkedIn, GitHub, website, or portfolio for each other unless CUSTOM RULES explicitly say they are interchangeable.
- EEO and demographics:
  - Use stored preferences first.
  - Else select `Prefer not to say` when available.
  - Else ask the user.
- "How did you hear about us?":
  - Prefer `Job Board`, `Online`, `Company Website`, or the closest visible equivalent.
  - Ask the user if none of those are available and the field is required.
- Free text:
  - If `free_text_mode=ask`, ask the user for required free-text questions outside simple availability/location prompts.
  - If `free_text_mode=best_effort`, answer safe motivation questions using only candidate facts, visible job context, and the staged cover letter text.
  - Always ask the user for technical screening, architecture/design claims, legal statements, security-clearance questions, or anything that would require invention.

## Consent Rules

- Apply the configured policy for:
  - marketing opt-in
  - SMS consent
  - talent pool consent
- Always auto-accept:
  - required terms and conditions
  - privacy acknowledgement
  - accuracy certification
- Never auto-accept optional marketing or outreach consent unless the configured policy says `auto_allow`.
- If a consent field does not match any known category, ask the user when it is required and skip it when it is optional.

## Prefill Review

- `prefill_review_mode=off`:
  - trust visible prefills unless they cause validation errors
- `prefill_review_mode=smart`:
  - verify name, email, phone, location, parsed work-history dates, parsed education dates, and any required parsed field
- `prefill_review_mode=full`:
  - review every visible prefilled field before advancing

## Review And Submit

1. When the final review step or final submit button is visible, emit:
   - `emit_events=[{"name":"application_review_reached","data":{"review_visible":true}}]`
2. Call `ask_user` before submitting.
   - Use options like `["Submit", "Don't submit"]`.
3. If the user declines, emit:
   - `emit_events=[{"name":"application_cancelled","data":{"reason":"user declined final submit"}}]`
   then call `think(next_action="done")`.
4. If the user approves, click the final submit button and emit:
   - `emit_events=[{"name":"application_submit_attempted","data":{"submit_button_label":"<visible label>"}}]`
5. After clicking submit once, wait for confirmation, redirect completion, or validation errors.
   - Do not click the submit button again unless the page clearly returns to an editable validation-error state on the same review step.
6. If confirmation is visible, emit:
   - `emit_events=[{"name":"application_submitted","data":{"confirmation_detected":true,"confirmation_text":"<short visible confirmation text>"}}]`
   then report completion and call `think(next_action="done")`.
7. If submit returns validation errors, fix them and continue.

## Recovery

- If `open_url`, Apply, Continue, or email entry redirects to a different site before the form appears:
  - treat the redirect as expected
  - wait for the destination page to settle
  - re-detect whether it is the form, auth gate, restricted-access page, builder-only flow, manual follow-up, or unsupported page
- If navigation leaves the application flow unexpectedly:
  - emit `application_error`
  - use `go_back` or `open_url` to recover
- If the session times out:
  - emit `application_auth_required`
  - ask the user to refresh or sign in again
- If Cloudflare, "Checking your browser", rate-limit pages, or similar anti-bot interstitials appear:
  - emit `application_auth_required`
  - ask the user to clear the blocker, then re-detect the page
- If the site opens the form in a new tab:
  - switch to the new tab and continue
- If file upload fails:
  - retry once
  - if it still fails, emit `application_error` and ask the user
- If the portal reaches an assessment, scheduler, or already-applied page:
  - use `references/manual-followup-guide.md`
  - emit `application_manual_followup_required`
  - explain the next step clearly
- If the portal forces a resume or cover-letter builder/import service with no path to use the staged files:
  - emit `application_failed`
  - call `flag`
  - call `think(next_action="done")`
- If the portal visibly states that the role is restricted to premium/member/subscriber/special access:
  - emit `application_failed`
  - call `flag`
  - call `think(next_action="done")`
- If the flow becomes unrecoverable:
  - emit `application_failed`
  - call `flag`
  - call `think(next_action="done")`
