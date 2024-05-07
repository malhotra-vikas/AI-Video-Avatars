"""
Run this example: beam run scraper.py:scrape_site
"""
from beam import App, Runtime, Image

import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = App(
    name="web-scraper",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        image=Image(
            python_version="python3.8",
            python_packages=["bs4", "transformers", "torch"],
        ),
    ),
)


@app.run()
def scrape_site():
    url = "https://wcca.wicourts.gov/caseDetail.html?caseNo=2024TR010857&countyNo=40&index=0&isAdvanced=true&mode=details#charges"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all <div> elements
    div_elements = soup.find_all('div')
    
    # Print the list of <div> elements
    for i, div in enumerate(div_elements, start=1):
        print(f"Div {i}: {div}")
