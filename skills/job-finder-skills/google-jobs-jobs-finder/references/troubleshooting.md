# Google Jobs — Troubleshooting

## Captchas
- Google may show a captcha if it detects automated browsing
- Signs: a challenge page replaces the search results, often with "I'm not a robot" or image selection
- Action: call `ask_user` with `["Done"]` so the human can solve it
- If `ask_user` fails or the captcha persists, `flag` and end the mission

## Sign-In Redirects
- Google sometimes redirects to a sign-in page (accounts.google.com)
- This can happen mid-session, especially after clicking apply links or after idle time
- Action: call `go_back` first. If that doesn't return to Jobs, re-open the search URL with `&udm=8`
- Do not attempt to sign in

## Empty Results
- Some queries return no job listings at all
- Signs: the Jobs page loads but the left panel has no cards, or shows a "no results" message
- Action: report that no listings were found for the query and end cleanly
- This is not an error — it's a valid outcome

## Stale Card State After Scroll
- After scrolling the left panel, previously visible cards may shift or be replaced
- The DOM updates asynchronously — cards that were at position N may now be at position M
- Never click based on a pre-scroll element index. Always re-read the visible cards after any scroll.

## Duplicate Cards
- The same job can appear multiple times in the card list (Google sometimes shows duplicates)
- `process_focused_job` handles deduplication — if it returns `duplicate=true`, just move on
- Two consecutive duplicates from the same visible set suggest you've seen all cards in this area — scroll for more

## Off-Page Navigation
- Clicking certain elements (company links, "more jobs from..." links) can navigate away from the Jobs page
- If the current URL no longer contains the search query or `udm=8`, you've left the Jobs page
- Recovery: `go_back`, then verify you're on the Jobs page. If not, re-open the URL.

## Search Context Drift
- Google can keep you inside the Jobs UI but silently switch the query context to a different search such as `Jobs at <company>`
- Signs: the page title or `q=` parameter no longer matches the original requested search query
- Action: treat this as recoverable workflow drift, not as a rejected job
- Recovery: prefer `go_back` if the previous page in navigation history was the original Google Jobs results page; otherwise re-open the original search URL with `&udm=8`

## Incomplete Job Details
- Some job cards load with partial details in the right panel (missing description, missing apply links)
- `process_focused_job` retries once automatically if required fields are missing
- If it still returns `processable=false`, skip that card and move on

## Regional Differences
- Google Jobs layout is consistent across regions, but available filters and card density may vary
- Some regions show salary info prominently, others don't
- The skill workflow doesn't change — `process_focused_job` handles whatever fields are available
