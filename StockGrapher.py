a# Install required packages
# Uncomment the lines below if you need to install the packages
# !pip install yfinance
# !pip install bs4
# !pip install nbformat
# !pip install matplotlib

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings
import matplotlib.pyplot as plt

# Ignore all warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def make_graph(stock_data, revenue_data, stock):
    """
    Create a graph showing stock price and revenue over time
    """
    stock_data_specific = stock_data[stock_data.Date <= '2021-06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Stock price
    axes[0].plot(pd.to_datetime(stock_data_specific.Date), 
                 stock_data_specific.Close.astype("float"), 
                 label="Share Price", 
                 color="blue")
    axes[0].set_ylabel("Price ($US)")
    axes[0].set_title(f"{stock} - Historical Share Price")

    # Revenue
    axes[1].plot(pd.to_datetime(revenue_data_specific.Date), 
                 revenue_data_specific.Revenue.astype("float"), 
                 label="Revenue", 
                 color="green")
    axes[1].set_ylabel("Revenue ($US Millions)")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"{stock} - Historical Revenue")

    plt.tight_layout()
    plt.show()


# ============================================================================
# TESLA DATA
# ============================================================================

# Get Tesla stock data
data = yf.Ticker("TSLA")
print(data)

# Get historical data
tesla_data = data.history(period="max")
print(tesla_data)

# Reset index to get Date as a column
tesla_data.reset_index(inplace=True)

# Get Tesla revenue data from web
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm"
blank = requests.get(url)
html_data = blank.content

# Parse HTML
soup = BeautifulSoup(html_data, 'html.parser')

# Extract revenue data
tesla_revenue = pd.DataFrame(columns=['Data', 'Revenue'])
print(tesla_revenue)

table = soup.find_all("tbody")[1]

for row in table.find_all("tr"):
    cols = row.find_all("td")
    if len(cols) != 2:
        continue
        
    date = cols[0].text.strip()
    revenue = cols[1].text.strip()
    
    tesla_revenue = pd.concat(
        [tesla_revenue, pd.DataFrame({"Date": [date], "Revenue": [revenue]})],
        ignore_index=True
    )

print(tesla_revenue)

# Clean revenue data - remove commas and dollar signs
tesla_revenue["Revenue"] = tesla_revenue['Revenue'].str.replace(',|\\$', "", regex=True)
print(tesla_revenue)

# Remove empty revenue entries
print(tesla_revenue)
tesla_revenue = tesla_revenue[tesla_revenue['Revenue'] != ""]
print(tesla_revenue)

# Display last 5 rows
print(tesla_revenue.tail(5))

# Create Tesla graph
make_graph(tesla_data, tesla_revenue, "Tesla")


# ============================================================================
# GAMESTOP DATA
# ============================================================================

# Get GameStop stock data
data2 = yf.Ticker("GME")
print(data2)

# Get historical data
gme_data = data2.history(period="max")

# Reset index
gme_data.reset_index(inplace=True)
print(gme_data.head(5))

# Get GameStop revenue data from web
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html"
html_data2 = requests.get(url).content

# Parse HTML
soup2 = BeautifulSoup(html_data2, "html.parser")

# Extract revenue data
gme_revenue = pd.DataFrame(columns=['Date', 'Revenue'])
table = soup2.find_all("tbody")[1]

for row in table.find_all("tr"):
    cols = row.find_all("td")
    if len(cols) != 2:
        continue
        
    date = cols[0].text.strip()
    revenue = cols[1].text.strip()
    
    gme_revenue = pd.concat(
        [gme_revenue, pd.DataFrame({"Date": [date], "Revenue": [revenue]})],
        ignore_index=True
    )

print(gme_revenue)

# Clean revenue data
gme_revenue["Revenue"] = gme_revenue['Revenue'].str.replace(',|\\$', "", regex=True)

# Remove empty revenue entries
gme_revenue = gme_revenue[gme_revenue['Revenue'] != ""]

# Display last 5 rows
print(gme_revenue.tail(5))

# Create GameStop graph
make_graph(gme_data, gme_revenue, "GameStop")
