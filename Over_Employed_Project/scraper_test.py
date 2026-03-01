from seek_scraper import get_seek_listings
from indeed_scraper import get_indeed_listings

results = get_seek_listings(
    'https://www.seek.com.au/gym-jobs/in-brisbane',
    debug=True   # <-- add this
)
print(results)

results = get_indeed_listings(
    'https://au.indeed.com/jobs?q=supplements&l=brisbane+qld&from=searchOnHP&vjk=16aaf42053b9249f',
    debug=True
)
print(results)