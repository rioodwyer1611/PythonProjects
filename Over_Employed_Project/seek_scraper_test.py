from seek_scraper import get_seek_listings

results = get_seek_listings(
    'https://www.seek.com.au/gym-jobs/in-brisbane',
    debug=True   # <-- add this
)
print(results)