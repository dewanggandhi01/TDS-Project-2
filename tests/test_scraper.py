import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.data_loader import scrape_wikipedia, clean_scraped_df

task = "Scrape the list of highest grossing films from Wikipedia. It is at the URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
df = scrape_wikipedia(task)
df = clean_scraped_df(df)
print(df.head())
print(df.dtypes)
