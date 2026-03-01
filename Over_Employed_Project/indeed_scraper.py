"""
pip install playwright
playwright install chromium
"""

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

"""
Function takes a SEEK url and returns all the job listings given on the first page.
Input: url - A string referring to a specific seek url search.
Output: A dictionary containing all extracted job listings from SEEK.

Each listing looks like:
<li class="css-1ac2h1w eu4oa1w0">
    <div class="cardOutline tapItem dd-privacy-allow result job_cedaff16c1967440 resultWithShelf sponTapItem desktop vjs-highlight css-u6kqdl eu4oa1w0">
        <div class="slider_container css-weo834 eu4oa1w0" data-testid="slider_container">
            <div class="slider_list css-1bej0z4 eu4oa1w0">
                <div data-testid="slider_item" class="slider_item css-17bghu4 eu4oa1w0">
                    <div data-testid="fade-in-wrapper" class="css-u74ql7 eu4oa1w0">
                        <div class="job_seen_beacon">
                            <table class="mainContentTable css-131ju4w eu4oa1w0" cellpadding="0" cellspacing="0" role="presentation">
                                <tbody>
                                    <tr>
                                        <td class="resultContent css-1o6lhys eu4oa1w0">
                                            <div class="css-pt3vth e37uo190">
                                                <h2 class="jobTitle css-1o1rnx9 eu4oa1w0" tabindex="-1">
                                                <a id="job_cedaff16c1967440" data-mobtk="1jikeam56ggnd804" data-jk="cedaff16c1967440" data-hiring-event="false" target="_blank" data-hide-spinner="true" role="button" aria-label="full details of Sales Assistant" class="jcs-JobTitle css-1baag51 eu4oa1w0" href="/rc/clk?jk=cedaff16c1967440&amp;from=hp&amp;tk=1jikeam56ggnd804&amp;bb=l5BECWqecybscUC5jqPs25Gv6n3x0C4j2KXojmlKpyQN4PBj19XtJEUJVHwEsP6ZwXH867KQuVPvZ1qLGQrvM7NQe8AqkhI5IzyinRFyZWc8EtEIFiUQRxy9WW1rA1qRvgACV9GsnBt39MKkS1dYmg%3D%3D&amp;xkcb=SoAL67M3mUUlC_S0zB0LbzkdCdPP" aria-pressed="true">
                                                    <span title="Sales Assistant" id="jobTitle-cedaff16c1967440">
                                                        Sales Assistant
                                                    </span>
                                                </a>
                                                </h2>
                                                <div class="css-j160pq e37uo190">
                                                    <div class="mosaic-provider-jobcards-1uo542s e1xnxm2i0">
                                                        <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                            <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">Often responds within 3 days
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="css-u74ql7 eu4oa1w0">
                                                <div class="company_location css-1k93hyy e37uo190">
                                                    <div data-testid="timing-attribute">
                                                        <div class="css-1afmp4o e37uo190">
                                                            <span data-testid="company-name" class="css-19eicqx eu4oa1w0">
                                                                Supps247 Richlands
                                                            </span>
                                                        </div>
                                                        <div data-testid="text-location" class="css-1f06pz4 eu4oa1w0">
                                                            Richlands QLD 4077
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="jobMetaDataGroup css-jf723e eu4oa1w0">
                                                    <ul class="heading6 tapItem-gutter metadataContainer css-1hl5lcb eu4oa1w0">
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0" data-testid="attribute_snippet_testid">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">Full-time
                                                                        <span class="css-12dr2u9 eu4oa1w0">
                                                                             +1
                                                                        </span>
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0" data-testid="attribute_snippet_testid">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">
                                                                        Weekend availability
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">
                                                                        Free drinks
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">
                                                                        Professional development assistance
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">
                                                                        Employee discount
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">
                                                                        Relocation assistance
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                        <li class="mosaic-provider-jobcards-fswglz e1xnxm2i0">
                                                            <div class="mosaic-provider-jobcards-1oa1vqn ecydgvn0">
                                                                <div class="mosaic-provider-jobcards-1f1q1js ecydgvn1">
                                                                    <span class="css-zydy3i e1wnkr790">
                                                                        Visa sponsorship
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </li>
                                                    </ul>
                                                    <div class="heading6 error-text tapItem-gutter">
                                                    </div>
                                                </div>
                                                <div role="presentation" class="css-r19t1s eu4oa1w0">
                                                    <div class="css-6stls4 eu4oa1w0">
                                                        <span class="iaIcon css-n8u52n e1wnkr790" data-testid="indeedApply">
                                                            <svg xmlns="http://www.w3.org/2000/svg" focusable="false" role="img" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true" class="mosaic-provider-jobcards-1moe62j eac13zx0">
                                                                <path d="M4.406 19.533a1.103 1.103 0 01-1.082-.097c-.34-.221-.51-.54-.51-.954v-4.314L11.09 12 2.815 9.833V5.518c0-.414.17-.732.51-.954.34-.222.7-.254 1.08-.096l15.341 6.482c.469.207.703.557.703 1.05s-.234.844-.703 1.05l-15.34 6.483z">
                                                                </path>
                                                            </svg>
                                                            <span>
                                                                Easily apply
                                                            </span>
                                                        </span>
                                                    </div>
                                                    <div class="underShelfFooter">
                                                        <div class="heading6 tapItem-gutter css-1wcu7u6 eu4oa1w0">
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                            <div class="ctaContainer ctaContainer_withCompanyInfo">
                                <button class="bookmark bookmark-tap-target mosaic-provider-jobcards-ykqx5t e8ju0x50" aria-label="Save job Toggle" aria-pressed="false">
                                    <svg xmlns="http://www.w3.org/2000/svg" focusable="false" role="presentation" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true" class="mosaic-provider-jobcards-s0xhw4 eac13zx0">\
                                        <path d="M12 18.221l-4.027 1.723c-.758.319-1.477.256-2.157-.19-.68-.446-1.02-1.077-1.02-1.892V5.072c0-.63.222-1.166.666-1.61.443-.443.98-.665 1.61-.665h9.856c.63 0 1.166.222 1.61.665.443.444.665.98.665 1.61v12.79c0 .815-.34 1.446-1.02 1.892-.679.445-1.398.509-2.156.19L12 18.22zm0-2.493l4.928 2.115V5.072H7.072v12.77L12 15.729z">
                                        </path>
                                    </svg>
                                </button>
                                <button class="dislike-tap-target mosaic-provider-jobcards-1pc3wcg e8ju0x50" aria-label="Not interested" data-testid="dislikeicon">
                                    <svg xmlns="http://www.w3.org/2000/svg" focusable="false" role="img" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true" class="mosaic-provider-jobcards-1jau22k eac13zx0">
                                        <path d="M3.126 15.88c-.601 0-1.13-.228-1.589-.686-.458-.458-.686-.987-.686-1.588V11.97c0-.126.016-.262.05-.406.033-.144.07-.277.112-.398l3.035-7.151c.17-.381.455-.704.855-.967.4-.263.815-.395 1.247-.395h11.035v13.228l-5.725 5.68a1.826 1.826 0 01-1.028.534 1.815 1.815 0 01-1.128-.182 1.664 1.664 0 01-.783-.795 1.533 1.533 0 01-.094-1.093l1.041-4.144H3.126zM15 14.828V4.928H6.15L3.138 11.97v1.636h9.155l-1.278 5.207L15 14.827zm5-12.174c.63 0 1.166.222 1.61.666.443.443.665.98.665 1.61v8.677c0 .63-.222 1.166-.665 1.61-.444.443-.98.665-1.61.665h-2.815v-2.275H20V4.928h-2.815V2.653H20z">
                                        </path>
                                    </svg>
                                </button>
                            </div>
                            <div aria-live="polite">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <span aria-live="polite" class="visually-hidden css-16euvrx eu4oa1w0">
    </span>
</li>
"""
STEALTH_SCRIPTS = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-AU', 'en'] });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });
    window.chrome = { runtime: {} };
    Object.defineProperty(Notification, 'permission', { get: () => 'default' });
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {
        if (parameter === 37445) return 'Intel Inc.';
        if (parameter === 37446) return 'Intel Iris OpenGL Engine';
        return getParameter.call(this, parameter);
    };
"""


def get_indeed_listings(url: str, debug: bool = False) -> list[dict]:
    """
    Takes an Indeed search URL and returns all job listings from the first page.

    Args:
        url:   An Indeed search URL e.g. "https://au.indeed.com/jobs?q=gym&l=Brisbane+QLD"
        debug: If True, saves raw HTML to debug_indeed.html for inspection
    """

    html = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )

        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            locale="en-AU",
            timezone_id="Australia/Brisbane",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        # Inject stealth patches before any page script runs
        context.add_init_script(STEALTH_SCRIPTS)

        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(4000)
        html = page.content()

        browser.close()

    if debug:
        with open("debug_indeed.html", "w") as f:
            f.write(html)
        print("DEBUG: HTML saved to debug_indeed.html — open in browser to inspect")

    # ── Parse ──────────────────────────────────────────────────────────────────
    soup = BeautifulSoup(html, "html.parser")
    job_cards = soup.find_all("div", class_="cardOutline")

    if debug:
        print(f"DEBUG: Found {len(job_cards)} cardOutline elements")

    jobs = []

    for card in job_cards:

        # Skip non-job cards (profile prompts etc.) — real cards have <h2 class="jobTitle">
        if not card.find("h2", class_="jobTitle"):
            continue

        job = {}

        # Title — from the title attribute on the <span> inside the job link
        title_anchor = card.find("a", class_="jcs-JobTitle")
        if title_anchor:
            title_span = title_anchor.find("span", title=True)
            job["title"] = title_span["title"] if title_span else title_anchor.get_text(strip=True)
        else:
            job["title"] = None

        # Job ID & URL — data-jk on the title anchor
        if title_anchor:
            job_key = title_anchor.get("data-jk")
            job["job_id"] = job_key
            job["url"] = f"https://au.indeed.com/viewjob?jk={job_key}" if job_key else None
        else:
            job["job_id"] = None
            job["url"] = None

        # Company — <span data-testid="company-name">
        company_tag = card.find("span", attrs={"data-testid": "company-name"})
        job["company"] = company_tag.get_text(strip=True) if company_tag else None

        # Location — <div data-testid="text-location">
        location_tag = card.find("div", attrs={"data-testid": "text-location"})
        job["location"] = location_tag.get_text(strip=True) if location_tag else None

        # Metadata — job type, salary, perks all in <ul class="metadataContainer">
        job_type = None
        salary = None
        metadata_items = []

        metadata_ul = card.find("ul", class_="metadataContainer")
        if metadata_ul:
            for li in metadata_ul.find_all("li"):
                text = li.get_text(strip=True)
                if not text:
                    continue
                if "salary-snippet-container" in li.get("class", []):
                    salary = text
                elif li.get("data-testid") == "attribute_snippet_testid" and job_type is None:
                    job_type = text.split("+")[0].strip()
                else:
                    metadata_items.append(text)

        job["job_type"] = job_type
        job["salary"] = salary
        job["perks"] = metadata_items if metadata_items else None

        # Response time — e.g. "Often responds within 3 days"
        response_tag = card.select_one(".mosaic-provider-jobcards-1uo542s")
        job["response_time"] = response_tag.get_text(strip=True) if response_tag else None

        # Easy apply flag
        job["easy_apply"] = bool(card.find(attrs={"data-testid": "indeedApply"}))

        jobs.append(job)

    return jobs