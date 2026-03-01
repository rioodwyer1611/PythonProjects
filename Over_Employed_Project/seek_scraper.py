"""
pip install playwright
playwright install chromium
"""

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

"""
Function takes a SEEK url and returns all the job listings given on the first page.
Input: url - A string referring to a specific seek url search.
Output: A dictionary containing all extracted job listings from SEEK.

Title looks like:

<h3 class="_1ybl4650 _6wfnkx4x losivq0 losivq3 losivq1t losivq8 _1rtxcgx4">
    <div class="_1ybl4650 _6wfnkx5f _6wfnkx51">
        <a href="/job/90462701?type=standard&amp;ref=search-standalone&amp;
        origin=cardTitle#sol=d556b83c901add2582d6d138c89e37d10069e7fa" 
        class="_1ybl4650 _1ybl465g _1ybl4658  _1ybl4650 _1ybl465g _1ybl4658 xtd45ud xtd45uf" 
        id="job-title-90462701" 
        data-automation="jobTitle" data-testid="job-card-title" 
        data-run-click-only="true" target="_self">
        Assistant Club Manager - Snap Fitness Morningside
        </a>
    </div>
</h3>

Company Looks Like:

<div class="_1ybl4650 _6wfnkx7t">
    <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1t losivq8 _1rtxcgx4">
        <div class="_1ybl4650 _6wfnkx5f _6wfnkx51">
            <span class="_1ybl4650 _6wfnkx5h _6wfnkx0 w4lx9z0">at 
            </span>
            <a href="/Snap-Fitness-jobs" 
            class="_1ybl4650 _1ybl465g _1ybl4658  _1ybl4650 _1ybl465g _1ybl4658 lys90e0 lys90e1" 
            aria-label="Jobs at Snap Fitness" title="Jobs at Snap Fitness" data-type="company" 
            data-automation="jobCompany" target="_self">
            Snap fitness Morningside
            </a>
        </div>
    </span>
</div>

Location, and Role type Looks like:

<div class="_1ybl4650 _6wfnkx7x _6wfnkx82">
    <div class="_1ybl4650 _6wfnkx5h _6wfnkx0 w4lx9z0">
        <p class="_1ybl4650">This is a Part time job
        </p>
    </div>
    <div class="_1ybl4650 _6wfnkx59 _6wfnkxhh _6wfnkx6p">
        <div class="_1ybl4650">
            <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1t losivq6 _1rtxcgx4">
                <span class="_1ybl4650" data-automation="jobCardLocation">
                    <a href="/gym-jobs/in-Morningside-QLD-4170" 
                    class="_1ybl4650 _1ybl465g _1ybl4658  _1ybl4650 _1ybl465g _1ybl4658 lys90e0 lys90e2" 
                    aria-label="Limit results to Morningside" tabindex="-1" 
                    title="Limit results to Morningside" data-type="location" 
                    data-automation="jobLocation" 
                    target="_self">Morningside
                    </a>
                </span>
                <span class="_1ybl4650" data-automation="jobCardLocation"> 
                    <a href="/gym-jobs/in-All-Brisbane-QLD" 
                    class="_1ybl4650 _1ybl465g _1ybl4658  _1ybl4650 _1ybl465g _1ybl4658 lys90e0 lys90e2" 
                    aria-label="Limit results to Brisbane QLD" tabindex="-1" 
                    title="Limit results to Brisbane QLD" data-type="location" 
                    data-automation="jobLocation" target="_self">Brisbane QLD
                </a>
                </span>
            </span>
        </div>
    </div>
</div>

Job Description First Part:
<div class="_1ybl4650 _6wfnkx4t _6wfnkx4y">
    <ul class="_1ybl4650 _1ybl4653 _6wfnkx59 _6wfnkxhh _6wfnkx6p _6wfnkxi9">
        <li class="_1ybl4650 _6wfnkx59">
            <div class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1t losivq6 _1rtxcgx4">
                <div class="_1ybl4650 _6wfnkx59 _6wfnkxgl _6wfnkx4 ad03uq1" aria-hidden="true">
                    <div class="_1ybl4650 _6wfnkx5x tpjjbi0 tpjjbi2">
                    </div>
                </div>
            </div>
            <div class="_1ybl4650 _6wfnkxr _6wfnkxp _6wfnkxb9">
                <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq6 _1rtxcgx4">
                Guaranteed Income with Business Growth Potential
                </span>
            </div>
        </li>
        <li class="_1ybl4650 _6wfnkx59">
            <div class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1t losivq6 _1rtxcgx4">
                <div class="_1ybl4650 _6wfnkx59 _6wfnkxgl _6wfnkx4 ad03uq1" aria-hidden="true">
                    <div class="_1ybl4650 _6wfnkx5x tpjjbi0 tpjjbi2">
                    </div>
                </div>
            </div>
            <div class="_1ybl4650 _6wfnkxr _6wfnkxp _6wfnkxb9">
                <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq6 _1rtxcgx4">
                Career and Training Development
                </span>
            </div>
        </li>
        <li class="_1ybl4650 _6wfnkx59">
            <div class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1t losivq6 _1rtxcgx4">
                <div class="_1ybl4650 _6wfnkx59 _6wfnkxgl _6wfnkx4 ad03uq1" aria-hidden="true">
                    <div class="_1ybl4650 _6wfnkx5x tpjjbi0 tpjjbi2">
                    </div>
                </div>
            </div>
            <div class="_1ybl4650 _6wfnkxr _6wfnkxp _6wfnkxb9">
                <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq6 _1rtxcgx4">
                Vibrant Culture and Gym Perks
                </span>
            </div>
        </li>
    </ul>
</div>

Description Part 2:
<div class="_1ybl4650 _6wfnkx4t _6wfnkx4y">
    <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq6 _1rtxcgx4" 
    data-testid="job-card-teaser" data-automation="jobShortDescription">
    Join Snap Fitness Morningside as an Assistant Club Manager—grow your fitness career while supporting sales, 
    team culture, and member experience.
    </span>
</div>

Job Age and General Field:
<div class="_1ybl4650">
    <div class="_1ybl4650 m6j21s0 m6j21s1" data-testid="job-classification" style="max-height: 35px; overflow: hidden;">
        <div class="_1ybl4650 _6wfnkxdl">
            <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq6 _1rtxcgx4">
                <div class="_1ybl4650 _6wfnkx59 _6wfnkxh9 _6wfnkxgh _6wfnkx7h _6wfnkxhp jo1oa00">
                    <span class="_1ybl4650 _6wfnkx5h _6wfnkx0 w4lx9z0">subClassification: Management
                    </span>
                    <span class="_1ybl4650" data-type="subClassification" data-automation="jobSubClassification">
                    Management
                    </span>
                    <div class="_1ybl4650 _6wfnkxfh">
                    </div>
                    <span class="_1ybl4650 _6wfnkx5h _6wfnkx0 w4lx9z0">
                    classification: Sport &amp; Recreation
                    </span>
                    <span class="_1ybl4650" data-type="classification" data-automation="jobClassification">
                    (Sport &amp; Recreation)
                    </span>
                </div>
            </span>
        </div>
    </div>
    <div class="_1ybl4650 _6wfnkx59 _6wfnkxh9 _6wfnkxgt _6wfnkxgp _6wfnkxn guyvzu2j">
        <div class="_1ybl4650 _6wfnkx4x _6wfnkxr _6wfnkxp _6wfnkxi5 _6wfnkxbx">
            <div class="_1ybl4650 _6wfnkx4x _6wfnkx4u">
                <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq4 _1rtxcgx4" data-automation="jobListingDate">
                9d ago
                </span>
            </div>
            <div class="_1ybl4650 _6wfnkx4t _6wfnkx4y">
                <span class="_1ybl4650 _6wfnkx4x losivq0 losivq1 losivq1u losivq6 _1rtxcgx4" data-automation="jobListingDate">
                9d ago
            </span>
        </div>
    </div>
</div>
"""
def get_seek_listings(url: str, debug: bool = False) -> list[dict]:
    with sync_playwright() as p:

        # ── Launch with stealth settings ───────────────────────────────────────
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",  # hides automation flag
                "--no-sandbox",
            ]
        )

        # Use a realistic viewport and locale
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

        page = context.new_page()

        # Hide the webdriver property that sites use to detect bots
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # ── Navigate and wait ──────────────────────────────────────────────────
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Give JS extra time to render the job cards after DOM loads
        page.wait_for_timeout(5000)

        html = page.content()

        # ── Debug: save HTML so you can see what SEEK actually returned ────────
        if debug:
            with open("debug.html", "w") as f:
                f.write(html)
            print("DEBUG: Page HTML saved to debug.html — open it in your browser to inspect")

        browser.close()

    # ── Parse ──────────────────────────────────────────────────────────────────
    soup = BeautifulSoup(html, "html.parser")
    job_cards = soup.find_all("article", attrs={"data-automation": "normalJob"})

    if debug:
        print(f"DEBUG: Found {len(job_cards)} job cards in the HTML")

    jobs = []

    for card in job_cards:
        job = {}

        title_tag = card.find("a", attrs={"data-automation": "jobTitle"})
        job["title"] = title_tag.get_text(strip=True) if title_tag else None

        job["url"] = (
            "https://www.seek.com.au" + title_tag["href"]
            if title_tag and title_tag.get("href") else None
        )

        company_tag = card.find("a", attrs={"data-automation": "jobCompany"})
        job["company"] = company_tag.get_text(strip=True) if company_tag else None

        location_tags = card.find_all("a", attrs={"data-automation": "jobLocation"})
        job["location"] = (
            ", ".join(t.get_text(strip=True) for t in location_tags)
            if location_tags else None
        )

        job_type_tag = card.find("p")
        if job_type_tag:
            raw = job_type_tag.get_text(strip=True)
            job["job_type"] = raw.replace("This is a ", "").replace(" job", "").strip()
        else:
            job["job_type"] = None

        bullet_points = []
        for li in card.find_all("li"):
            spans = li.find_all("span", recursive=True)
            if spans:
                text = spans[-1].get_text(strip=True)
                if text:
                    bullet_points.append(text)
        job["bullet_points"] = bullet_points if bullet_points else None

        teaser_tag = card.find("span", attrs={"data-automation": "jobShortDescription"})
        job["description"] = teaser_tag.get_text(strip=True) if teaser_tag else None

        sub_class_tag = card.find("span", attrs={"data-automation": "jobSubClassification"})
        job["sub_classification"] = sub_class_tag.get_text(strip=True) if sub_class_tag else None

        class_tag = card.find("span", attrs={"data-automation": "jobClassification"})
        job["classification"] = class_tag.get_text(strip=True).strip("()") if class_tag else None

        date_tag = card.find("span", attrs={"data-automation": "jobListingDate"})
        job["date_posted"] = date_tag.get_text(strip=True) if date_tag else None

        jobs.append(job)

    return jobs