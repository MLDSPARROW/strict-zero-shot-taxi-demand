# JOB 2: ONE dynamic page, https://miliroute.com/taxi-demand/ (Django)

Read `START_HERE.md` first (rules, tone, reporting). This file is the brief for the page. The owner decided: ONE page for all cities. The page must show the country first, then the city, and must be dynamic. It
must follow best practice for SEO, GEO (generative engine optimisation: being found, quoted and cited by AI answer systems) and AEO (answer engine optimisation: direct answers for search snippets and assistants).
The owner's goals: more visits to miliroute.com, and more visits to the owner's research paper and code.

## 0. Decisions already made (do not change them)

- One URL: `/taxi-demand/` (a subfolder of miliroute.com, not a subdomain: it shares the authority of the main site and needs no new DNS, certificate or host setup).
- NO page per city, per country or per area (thin-content risk and not wanted). The visitor's choice (country, city, weekday, time, area) is kept in the URL hash (`#australia/sydney-nsw/fri/22:00`), written with
  `history.replaceState`, never in the query string. The hash never reaches the server, so there is exactly one indexable URL.
- Data files (JSON, CSV) are files, not pages; they get `X-Robots-Tag: noindex` and are still crawlable.
- The data are files, not database tables: no migrations. No LLM calls at run time; every sentence is built from the data by fixed templates.
- English only. `<html lang="en">`. The explorer formats clock times by country: US, Canada, Australia 12-hour ("10:00 PM"), UK 24-hour; areas in km2 (square miles for the US).
- If the single page cannot rank for every city name, that is accepted. Do not add city pages. Make the page the best possible single page (sections 3, 6, 7).

## 1. How to work

- First read the Django project: Django and Python versions, apps, settings (static files, caching, security settings), base template and navigation, `urls.py`, an existing sitemap or robots.txt, the deployment, how
  tests run. Tell the owner in at most 15 lines. If something blocks you ask ALL questions in one message; otherwise carry on.
- New git branch `taxi-demand` in the website repository, small commits, no `Co-Authored-By` line and no "Generated with Claude Code" line anywhere (see START_HERE).
- Do not deploy, restart services, run migrations on production or touch the production database. Prepare code, tests and written deploy steps; the owner deploys.
- Reuse the site's look (base template, fonts, colours, header, footer). Change nothing outside this feature except the links and files named below.
- Start with the demo city (section 4). Sydney arrives in `results/sydney_*` of the research repository (`git pull`); more cities arrive from JOB 1 over days. The page must work with 1 city and with 100.

## 2. What the numbers are (hard rules for every word on the page)

- The numbers come from a research model (strict zero-shot transfer): it learned from taxi trips in Chicago, New York and San Francisco; for any other city it uses only the city's public map data (OpenStreetMap
  places), the calendar and the size of the airport. No taxi data of the city itself was used. They are model ESTIMATES and have not been checked against real trips in these cities.
- Say this in plain words near the top of the page (not only in the footer), and show "Data version" and "Last updated".
- Never write "real-time", "live", "measured" or "official". Never claim accuracy. Never make claims about waiting times, fares, driver earnings, surge pricing, safety or "the best place to earn".
  Say "expected taxi pickups (estimate)". "Taxi" means taxi-style pickups as recorded in the training cities, not rideshare only.
- The paper is not published yet: "Ghalejughi, M., Khattak, H. A., Hussain, Z. Spatial-scale-aware cross-city transfer for zero-shot taxi demand estimation (in preparation)". Settings `TAXI_DEMAND_PAPER_URL` and
  `TAXI_DEMAND_PAPER_DOI` (empty now): while empty, show "in preparation" and no link; once set, show the link and put the DOI in the structured data. Never invent a journal, DOI or result.
- The code is public: https://github.com/MLDSPARROW/strict-zero-shot-taxi-demand . Link it (this is the one link to the research that exists today).

## 3. The page, section by section (all server-rendered HTML except what is marked dynamic)

1. `<h1>`: "Taxi demand by city and time". A one-line subtitle: "Estimated taxi pickups by area, weekday and time in {N} cities in the United States, the United Kingdom, Canada and Australia" ({N} = published cities).
2. Answer box (AEO), 2 to 3 sentences built from `summaries.json`: for example the share of cities whose busiest half-hour of the week is on a Friday or Saturday night, the typical share of the ten busiest areas,
   and the cities where the airport area holds the largest share. Directly under it the estimate notice (section 2) and "Data version / Last updated".
3. "Choose a country and a city" (dynamic): step 1 country buttons (only countries with published cities); step 2 a city list for that country with a search box; then the explorer (section 5) and "Key facts" for the
   chosen city. The first visit preselects the country from the browser time zone (`Intl.DateTimeFormat().resolvedOptions().timeZone`: America/* -> United States or Canada, Europe/London -> United Kingdom,
   Australia/* -> Australia; no cookies, no IP lookup) and its largest city. The HTML sent by the server always contains the Key facts and the top-15 areas table of a fixed featured city (setting
   `TAXI_DEMAND_FEATURED_CITY`, default `sydney-nsw`), so that crawlers and visitors without JavaScript see real content. Never show crawlers different content from visitors.
4. "Compare cities": a real `<table>` with `<caption>` of ALL published cities, rows grouped by country, `<th scope="row">` = "City, Region, Country"; columns: busiest day and time, busiest area, share of the ten
   busiest areas, airport area share, number of areas. Every row has `id="c-<slug>"` and its city name is a link `href="#<country>/<slug>"` that loads the city into the explorer. Sortable by column with JavaScript;
   in HTML order by country then rank. Every row also carries one full sentence (a visually normal last column "In short") such as "Sydney, Australia: busiest on Friday around 10:00 PM, led by the airport area
   Mascot (about 10% of estimated demand)." These sentences are the passages that answer city-specific questions.
5. "What the cities have in common": 4 to 6 findings computed from `summaries.json` (a sentence each with the numbers), for example: in how many cities the busiest half-hour falls on a Friday or Saturday night; which five
   cities have the largest airport share; the five most concentrated cities (share of the ten busiest areas); how weekday mornings compare with weekend nights. Label them as model estimates.
6. "How the estimate is made" (3 short paragraphs in plain words) and "What it cannot tell you" (a short list: not real counts, not checked against local trips, taxi-style pickups not rideshare, yearly level not
   known, a typical week not a specific day, no weather or events).
7. FAQ, 10 to 12 questions, each with a direct 40 to 60 word answer (visible; the same text goes in the FAQPage structured data): What is taxi demand? When is taxi demand highest in cities? Which parts of a city have the
   most taxi pickups? Why is the airport area so busy in some cities? Are these real taxi counts? How was this estimated? How accurate is it? Does it include Uber or Lyft? Can I download or use the data? How often is it
   updated? Why only these cities? Who made this? Numeric answers come from the data.
8. "Data, code and paper": download links (cities list, one JSON and CSV per city: they are files, section 7), the GitHub link, the paper line (section 2), "Cite this page" with a copyable BibTeX and a plain text
   citation (authors, title, "Miliroute taxi demand estimates", data version, date, URL), the licence (`TAXI_DEMAND_LICENCE`, default "CC BY 4.0", the owner will confirm), the attribution lines (always "Map data (c)
   OpenStreetMap contributors (ODbL)", then the `attribution` of each country present).
9. Footer: last updated, data version, contact link of the site.

## 4. Inputs, the demo city, the import command

Folder `TAXI_DEMAND_DATA_DIR` (a setting; JOB 1 writes `miliroute/export/` of the research repository, copy or link it here; never commit real data to git, git-ignore the folder and document how to copy it to the
server):
- `cities.csv`, one row per city, columns: slug, name, country, country_slug (`united-states`, `united-kingdom`, `canada`, `australia`), region, area_label ("suburb", "ZIP code area", "ward", "postal-code area"; add "s"
  for the plural), lat, lon, timezone, locale, calendar_year, areas, median_area_km2, runs, heads, model_version, built_on, status (`published` or `draft`), reason, total_reliable, default_total_per_day,
  attribution, wikidata_id (optional).
- `cities/<country_slug>/<slug>.json` with the keys: `names[]`, `pid[]`, `area[]` (km2), `jobs[]`, `res[]` (may be null), `rhythm[7][48]` (share of the week's city trips by weekday, Monday = 0, and half-hour,
  0 = 00:00 to 00:30 local clock time; sums to 1), `nd[7]` (how many of each weekday the year has), `share_year[]` (sums to 1), `share_ws[area][7][48]` (the area's share of the city's demand in that weekday and
  half-hour, integers = share x 1,000,000), `paths[]`, `w`, `h`, `cx[]`, `cy[]` (SVG path of each area in map units, map size, label points), `labels[]` (optional, `[{"i": index, "text": "City"}]`; if missing label the 8
  areas with the highest `share_year`), `airport_i[]`, `centre_i[]` (optional lists of area indices), `level_year_strict` (a rough yearly total; probably far too low), `heads`, `runs`, `area_label`.
  Expected pickups of area i at weekday w, half-hour s, for a daily city total T: `v(i,w,s) = (365 * rhythm[w][s] / nd[w]) * share_ws[i][w][s] / 1e6` per trip a day of city total, `expected = T * v(i,w,s)`.
  Demand index = `100 * v(i,w,s) / max(v)` over all areas, weekdays and half-hours of the city.
- `cities/<country_slug>/<slug>_areas.csv` (rank, area, share_pct, jobs, residents, run_to_run_sd_pct), `summaries.json` (per city: busiest weekday and half-hour of the week, the three busiest areas with shares,
  share of the ten busiest, airport share, centre share, ratio of weekday mornings 7 to 9 to Friday and Saturday nights 22 to 02, quietest half-hour, areas, runs).
- Reference implementation of the explorer: `improve/sydney_page_template.html` (plain HTML, CSS, JavaScript, SVG; tested) and `improve/build_sydney_page.py` (how the map paths are made) in the research repository.
  Sydney's real file will be `results/sydney_week.json` plus the map keys of the builder. Port the template; do not rewrite it from scratch.
- Until real files exist, build and test with a small FAKE city generated in the same format (about 40 areas, named "demo-city", never published, `noindex` if shown) and a second fake city of another country.
  Real files must then work with no code change.
- Management command `import_taxi_demand`: validates the folder (section 9), prints a report, and writes the cached summaries so that no request ever opens a big city file.
- The daily total is the weakest number. Since `total_reliable` is false: the explorer shows the demand index and the area's share by default, and offers a control "Show expected pickups for a city total that I type"
  (empty by default; the rough value `level_year_strict / 365` may be shown as a hint with the warning "probably far too low"). Never show absolute pickups by default.

## 5. The explorer (dynamic, one page, all cities)

- Port the reference page into `static/taxi_demand/explorer.js` and `explorer.css` and a template partial. Load the chosen city's JSON with `fetch` after first paint, show a loading state, and on an error a message that
  points to the tables. Cache loaded cities in memory. Keep its behaviour: pick an area (map click, search, list); weekday buttons; half-hour slider with Play; map shaded by value; side card with the value, share, rank
  and the day's bar chart; the tab "Average per area" with a sortable table; colour scale "same all week" or "fits this half-hour". Start on the weekday and half-hour with the highest rhythm among Monday to Friday and on the
  area with the highest `share_year`.
- Replace the word "suburb" by the city's `area_label`. Add the demand-index mode (default) and the typed-total mode of section 4.
- Hash state: parse on load, update with `replaceState` on every change, `hashchange` support, so that a shared link restores the view. Back and forward buttons must not trap the visitor.
- Dark mode (`prefers-color-scheme` and the site's own theme), keyboard access to every control with a visible focus, aria labels, `prefers-reduced-motion`, works at 360 px width with no horizontal page scroll.
- No external libraries, fonts or trackers in this component. JavaScript at most 70 KB gzipped for the page.

## 6. SEO (you are the SEO expert here)

- `<title>` at most 60 characters, for example "Taxi Demand by City and Time ({N} Cities) | Miliroute"; meta description 140 to 155 characters, for example "See when and where taxi demand is highest in {N} cities in
  the US, UK, Canada and Australia. Free interactive map by area, weekday and time. Model estimates." ({N} is the number of published cities; update both automatically.) One `<h1>`, a logical `<h2>`/`<h3>` outline.
- Canonical (absolute, https, trailing slash). `robots` meta `index,follow`. `lang`. Mobile viewport. Open Graph and Twitter card with one 1200x630 image (a map of a published city made at build time from the JSON paths;
  command `build_taxi_demand_image`, Pillow or cairosvg; tell the owner about the new dependency).
- Sitemap: add `/taxi-demand/` with `lastmod` = the newest `built_on` to the site's sitemap (create a sitemap index if none exists; `django.contrib.sitemaps`). Add the sitemap line to `robots.txt`. Tell the owner to
  submit it in Google Search Console and Bing Webmaster Tools; never claim you did. Optional: an IndexNow key file and a command that pings it after a data update.
- Internal links: a link to `/taxi-demand/` in the main navigation or footer (ask the owner where if unclear) and from one or two related existing pages if they exist. Breadcrumb "Home > Taxi demand" (visible and as
  `BreadcrumbList`).
- Structured data, JSON-LD, every block must match what is visible on the page: `WebPage`; `WebApplication` (name, url, `applicationCategory` "UtilitiesApplication", `operatingSystem` "Any", free offer; do not invent
  ratings); one `Dataset` for the whole collection (name, description, url, license, creator = the three authors and Miliroute, `isAccessibleForFree`, keywords, `temporalCoverage` = `{calendar_year}-01-01/{calendar_year}-12-31`,
  `variableMeasured` = "estimated taxi pickups per half-hour", `distribution` = `DataDownload` for the cities file, `isBasedOn` = the GitHub repository as `SoftwareSourceCode`, `citation` = the paper once its URL or DOI is
  set, `hasPart` = one small `Dataset` per city with name "Taxi demand estimates for {City}, {Region}, {Country}", url `https://miliroute.com/taxi-demand/#<country>/<slug>`, `spatialCoverage` = `Place` with
  `GeoCoordinates` and `sameAs` the Wikidata URL when `wikidata_id` is set, and `distribution` of its JSON and CSV); `FAQPage` for the visible FAQ only (do not expect Google to show FAQ rich results; it helps other
  engines and AI systems); `BreadcrumbList`. Validate each block with `json.loads` in tests and tell the owner to check Google's Rich Results Test and the Schema.org validator.
- Content rules: no hidden text, no keyword stuffing, no text that only appears after a click or after JavaScript, no duplicate paragraphs, no pages for cities without data. The only repeated block is the table row
  structure.
- Speed: HTML at most 150 KB before compression, LCP under 2.5 s and CLS under 0.1 on mobile, images with width and height, city JSON fetched lazily (`<link rel="preload" as="fetch" crossorigin>` only for the
  featured city), static files versioned (`?v=<model_version>`) with `Cache-Control: public, max-age=31536000, immutable`, HTML cached for one hour, gzip or brotli for JSON, CSS and JS (tell the owner what to change in
  the web server if needed).

## 7. GEO and AEO (you are the AI expert here)

- Answer first: every section starts with its answer in one or two sentences; questions as headings; 40 to 60 word answers; short paragraphs (under 120 words), lists and tables instead of long prose. Every key sentence
  is self-contained: it names the city, region and country, gives the number with its unit, and says "estimate", so that one sentence copied alone is still true and clear.
- Provenance on the page and in the data: "Source: Miliroute taxi demand model v1, data version {X}, {date}"; stable anchors (`#c-<slug>`); a visible "Last updated"; a change log section or file.
- Machine-readable data, all with CORS `*`, `ETag`, `Cache-Control`, `Link: <licence URL>; rel="license"` and `X-Robots-Tag: noindex`:
  `/taxi-demand/api/v1/cities.json` (slug, name, country, region, lat, lon, areas, runs, model_version, built_on, page_url `https://miliroute.com/taxi-demand/#<country>/<slug>`, data_url),
  `/taxi-demand/api/v1/<country>/<slug>.json` (the city file), `/taxi-demand/api/v1/<country>/<slug>/areas.csv`, `/taxi-demand/api/v1/<country>/<slug>/week.csv.gz` (long format: area, weekday, half_hour, share),
  `/taxi-demand/api/v1/openapi.json` (OpenAPI 3 description).
- `/llms.txt` at the site root (extend it if it exists): what the page is, the honesty statement, links to the page, the API, the GitHub repository and the paper line; and `/taxi-demand/llms-full.txt`: plain text with the
  method in plain words, the limits, and for every published city a paragraph with its key facts (same sentences as the table). Both generated from the data.
- `robots.txt`: allow everything under `/taxi-demand/`. Name the search and answer crawlers as allowed (Googlebot, Bingbot, OAI-SearchBot, ChatGPT-User, Claude-SearchBot, Claude-User, PerplexityBot,
  Perplexity-User). Put the AI-training crawlers (GPTBot, ClaudeBot, Google-Extended, Applebot-Extended, CCBot) behind one setting `TAXI_DEMAND_ALLOW_AI_TRAINING` (default True). Check each vendor's current
  documentation for the exact names and tell the owner what you found.
- Entities: always "City, Region, Country" in full somewhere (London, United Kingdom is not London, Ontario); `wikidata_id` and `sameAs` when available (look up with the Wikidata API and accept only a result within
  30 km of the city's coordinates); the three authors as `Person` with links the owner provides (ORCID, Google Scholar, GitHub); the organisation Miliroute as publisher.
- Citable by design: the "Cite this page" block, a stable version string, a downloadable dataset with licence. Tell the owner (section 10) to archive the dataset and code on Zenodo for a DOI; add the DOI to the
  Dataset markup through a setting.

## 8. Quality: accessibility, security, performance

- WCAG 2.1 AA: contrast, labels, focus, table markup, a text alternative to the map (the tables), no colour-only meaning.
- City and weekday values from the hash or the API are validated against fixed lists (regex `^[a-z0-9-]+$`, known slugs); unknown values do nothing. No user data stored, no cookies, no new trackers; use the site's
  existing analytics if any, with its consent rules. CORS only on the API. No inline scripts except JSON-LD (so that a strict Content-Security-Policy works).
- Failure modes: JavaScript off (all tables and facts are there), a city file missing (message, the rest works), a very small screen, a city with no airport, a city with 1,000 areas (the list is searchable and
  virtualised or paged, the map stays fast).

## 9. Tests (Django tests; all must pass) and checks you run

- Import checks per city: for every weekday and half-hour the `share_ws` values sum to 1,000,000 within the number of areas; `rhythm` and `share_year` sum to 1 within 1e-6; the lists `names`, `area`, `paths`,
  `cx`, `cy`, `share_ws` have equal length; no NaN; slugs unique, `^[a-z0-9-]+$`; a published city has attribution, lat, lon, timezone and a country slug of the four.
- The page: status 200 with the demo cities; exactly one `<h1>`; title at most 60 characters; meta description 140 to 155 characters; one canonical with the exact URL; valid JSON-LD (every block `json.loads`, required
  keys present, `hasPart` count = number of published cities, FAQ in the markup = FAQ visible); every published city appears once in the table with its sentence; the estimate notice, "Data version" and "Last updated"
  are in the HTML; the words "real-time", "live", "measured", "official" and "guaranteed" never appear in generated text; no query-string URL is linked; HTML size under 150 KB with 100 fake cities.
- API: headers (CORS, ETag, Cache-Control, licence Link, X-Robots-Tag), valid CSV and gzip, OpenAPI file parses; unknown slug 404; `llms.txt` and `llms-full.txt` exist and list the published cities.
- Not unit tests, but do them: a screenshot of the page at 360 px and 1280 px (light and dark), a JavaScript-off check, a Lighthouse or equivalent run if the server has Chrome.

## 10. What the owner must do (put this list at the top of `docs/taxi_demand.md` in simple words)

1. Deploy the branch (your written steps), copy the data folder, run `import_taxi_demand`. 2. Add the link to the navigation. 3. Verify the site in Google Search Console and Bing Webmaster Tools; submit the
sitemap; ask Google to index `/taxi-demand/`. 4. Test with Google's Rich Results Test and PageSpeed Insights. 5. Create a GitHub release of the research repository and let Zenodo archive it (free DOI); send you the DOI
for the setting. 6. When a preprint or the paper exists: set `TAXI_DEMAND_PAPER_URL` and `TAXI_DEMAND_PAPER_DOI`. 7. Share the page where drivers, riders, city planners and data people read: a short post with three
findings from section 3.5, data communities, the paper's authors' networks. 8. Check Search Console queries each month and improve the FAQ and the findings from what people search. 9. After each data update: change
the data version, redeploy, ping IndexNow. 10. Confirm the licence, and ask the co-authors before the page goes public (the paper is not published yet).

## 11. Order of work and what to hand back

1. The short report of section 1. 2. Data loader, import command, demo cities, tests. 3. The page with all server-rendered sections. 4. The explorer. 5. SEO and structured data. 6. GEO/AEO files and the API.
7. `docs/taxi_demand.md`: architecture, data format, how to add cities (drop files, run the command), deploy and roll back, section 10, a manual checklist. 8. A final report in very simple English: what works, what
you could not check, what the owner must do.
Out of scope: pages per city, country or area, user accounts, comments, payments, ads, external map providers, live data, any LLM call at run time.
