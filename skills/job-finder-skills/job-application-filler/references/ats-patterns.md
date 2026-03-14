# ATS Patterns

Use these notes only after you know which platform is in front of you.

## Greenhouse

- Usually a clean single-page application.
- Login is uncommon.
- Resume upload often prefills the form immediately.
- Cover letter may be a file upload, a text area, or both.
- Review the parsed fields after upload.

## Lever

- Often single-page, but some employers split the flow into a few steps.
- Login is uncommon.
- Resume upload may appear near the top with custom styled controls.
- Follow the generic loop if the employer has customized the page heavily.

## Workday

- Often multi-step and usually account-driven.
- Login, registration, email verification, or password reset flows are common.
- Parsed work history and education are high-risk fields. Prefer `prefill_review_mode=smart` or `full`.
- Watch for repeated `Next`, `Save`, and `Review` buttons.

## LinkedIn Easy Apply

- Usually a modal or wizard flow.
- LinkedIn login is required.
- Resume upload and profile-based prefills are common.
- The flow can branch into assessments or recruiter questions. Treat those as manual follow-up if they leave the normal modal flow.

## BambooHR

- Usually simpler than Workday.
- Often a single page or a small number of steps.
- Resume and cover letter uploads are usually straightforward.

## SmartRecruiters

- Often multi-step with polished custom controls.
- Login may or may not be required.
- Look for review pages before the final submit button.

## Taleo

- Legacy layout and iframe-heavy pages are common.
- Expect awkward next-page transitions and session issues.
- If the flow becomes unstable, recover carefully and avoid duplicate submissions.

## ICIMS

- Often multi-step with employer-specific customization.
- Resume parsing can create bad date or title prefills.
- Watch for hidden required sections that appear only after certain answers.

## Generic Or Custom Portals

- Use the normal state machine from the main skill.
- Detect whether the page is still a job posting, an apply wizard, or a dead end.
- Prefer visible labels and required markers over assumptions.
