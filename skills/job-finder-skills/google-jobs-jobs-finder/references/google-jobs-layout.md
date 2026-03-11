# Google Jobs Page Layout

## URL Format
- Search: `https://www.google.com/search?q={query}&udm=8`
- The `&udm=8` parameter forces the Jobs vertical. Without it, results show web search.

## Page Structure
The Google Jobs interface is a two-panel layout:

### Left Panel — Job Cards List
- Scrollable list of job cards on the left side of the page (roughly left 45% of viewport)
- Each card shows: job title, company name, location, posted date, and sometimes salary
- Clicking a card loads its full details in the right panel
- The card list is a scroll container — new cards load as you scroll down
- When a card is selected/focused, it is visually highlighted

### Right Panel — Job Details
- Takes up roughly the right 55% of the viewport
- Shows full details for the currently selected card: title, company, location, description, apply buttons
- Apply buttons/links appear in this panel — they link to external sites (company careers, LinkedIn, Indeed, etc.)
- The detail panel updates when you click a different card in the left panel

## Scrolling
- The left panel is an independently scrollable container (not the page body)
- Use `scroll_container` targeting the left panel to load more cards
- `scroll_down` scrolls the entire page, which may not reveal more cards — use only as fallback
- After scrolling, the visible card list changes. Always re-read visible cards rather than relying on pre-scroll state.
- Scrolling may trigger lazy-loading of additional cards

## Filters
- Filter chips appear above the job cards (location, date posted, type, etc.)
- These are not typically needed for basic collection — the search query handles filtering

## Jobs Tab
- When you first load the search URL with `&udm=8`, the Jobs tab should be auto-selected
- If it's not selected, it appears as a tab/chip near the top of the search results
- If already highlighted/selected, do not click it again — this can reload the page
