from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

# URL of the webpage containing the files
url = "https://ita.ee.lbl.gov/html/contrib/WorldCup.html"

# Send a GET request to retrieve the webpage content
response = urlopen(url)

# Create a BeautifulSoup object to parse the HTML
soup = BeautifulSoup(response, "html.parser")

# Find all the links on the webpage
links = soup.find_all("a")

# Iterate over the links and download the files
for link in links:
    href = link.get("href")
    if href.endswith(".gz") and 'wc_day' in href:
        file_url = urljoin(url, href)
        file_name = href.split("/")[-1]
        
        # Send a GET request to download the file
        urlretrieve(file_url, file_name)
        
        print(f"Downloaded: {file_name}")