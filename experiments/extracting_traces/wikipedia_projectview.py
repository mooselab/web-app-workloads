import os
import csv
from urllib import request
from bs4 import BeautifulSoup

def extract_info(file_url):
    # Download the projectview file from wikipedia
    request.urlretrieve(file_url, 'file.txt')

    # Read file and count the hourly project views
    f = open('file.txt', 'r')
    lines = f.readlines()
    project_view = 0
    for line in lines:
        splits = line.split(' ')
        project_view += int(splits[2])

    # Define date_part and time_part with default values
    date_part = ""
    time_part = ""

    # Check if there are enough parts to split
    file_url_parts = file_url.split('-')
    if len(file_url_parts) == 4:
        date_part = file_url_parts[2]
        time_part = file_url_parts[3]

    # Rearrange the parts to the desired format
    formatted_date = f'{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:]}'

    # Write extracted info in a csv file
    writer.writerow([formatted_date, project_view])

    # Remove the file
    os.remove('file.txt')

 # Initializing the csv file
file = open('project_views_hourly_23.csv', mode='w', newline='', encoding='utf-8')  
writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['Time', 'Projectviews'])

def extract_urls(url, start_word):
    # Extract urls from the given webpage
    response = request.urlopen(url)
    if response.status == 200:
        html_content = response.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        links = soup.find_all('a')
        for link in links:
            href = link.get('href')
            if href.startswith(start_word):
                file_url = url + href
               
                # Extract month urls in the year webpage
                if start_word == '2023':  
                    print(file_url)              
                    extract_urls(file_url, 'projectviews')

                # Extract projectview urls from month webpage
                elif start_word == 'projectviews':
                    extract_info(file_url)

    else:
        print('Error occurred. Status code:', response.status)

# Change this URL for getting other years
url = 'https://dumps.wikimedia.org/other/pageviews/2023/'
extract_urls(url, '2023')