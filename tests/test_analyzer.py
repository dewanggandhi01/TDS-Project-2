import pandas as pd
from app.analyzer import analyze_scraped_data

def test_top_5_by_gross():
    df = pd.DataFrame({
        "Title": ["A", "B", "C", "D", "E", "F"],
        "Worldwide gross": [1, 2, 3, 4, 5, 6],
        "Year": [2011, 2012, 2013, 2014, 2015, 2016],
        "Rank": [1, 2, 3, 4, 5, 6],
        "Peak": [1, 2, 3, 4, 5, 6]
    })
    task = "List the top 5 films by worldwide gross."
    result = analyze_scraped_data(df, task)
    assert result[3] == ["F", "E", "D", "C", "B"]