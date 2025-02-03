import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm

#%% Data Scraping
# URL of the website
url1 = "https://markets.businessinsider.com/bonds/finder?p=1&borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
url2 = "https://markets.businessinsider.com/bonds/finder?p=2&borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"
url3 = "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&yield=&bondtype=2%2c3%2c4%2c16&coupon=&currency=184&rating=&country=19"


def get_data_from_web(url: str, verbose: bool = False) -> pd.DataFrame:
    data = {}
    # Send a GET request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table containing the bond data
        table = soup.find('table', {'class': 'table'})

        # Check if the table was found
        if table:
            # Extract the headers of the table
            headers = [header.text.strip() for header in table.find_all('th')]
            headers.append("ISIN")

            # Extract the rows of the table
            rows = []
            for row in table.find_all('tr')[1:]:  # Skip the header row
                # find ISIN
                first_cell = row.find('td')
                if first_cell:
                    link = first_cell.find('a')
                    if link:
                        # Extract the href attribute (hyperlink)
                        hyperlink = link.get('href')
                        ISIN = hyperlink.split("-")[-1].upper()
                        if verbose:
                            print("Hyperlink in the first cell of this row:", hyperlink)
                    else:
                        raise ValueError("No hyperlink in the first cell of this row.")
                else:
                    raise ValueError("No hyperlink in the first cell of this row.")

                # after finding the hyperlink, goes into its website and extracts
                response_data = requests.get("https://markets.businessinsider.com/" + hyperlink)

                soup_data = BeautifulSoup(response_data.content, 'html.parser')

                price = soup_data.find_all('span', 'price-section__current-value')[0].text

                table_data = soup_data.find('table', {'class': 'table table--no-vertical-border'})

                data_rows = []
                for row in table_data.find_all('tr')[1:]:  # Skip the header row
                    data_rows.append([cell.text.strip() for cell in row.find_all('td')])

                data[ISIN] = {
                    "ISIN": ISIN,
                    "price": price,
                    "coupon": data_rows[9][1],
                    "issue date": data_rows[7][1],
                    "maturity date": data_rows[14][1]
                }

            return pd.DataFrame(data).T
        else:
            raise ValueError("Table not found on the page.")
    else:
        raise ValueError(f"Failed to retrieve the webpage. Status code: {response.status_code}")

df_lst = []
for url in tqdm([url1, url2, url3]):
    df_lst.append(get_data_from_web(url))

#%% Data Pre-processing
df = pd.concat(df_lst, axis=0)
df.reset_index(drop=True, inplace=True)

# store as today's date
dt = datetime.now().strftime("%Y-%m-%d")
df.to_csv(f"{dt} bond yield.csv", index=False)