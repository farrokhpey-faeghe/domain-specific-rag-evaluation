"""WS1 (feat/data): corpus snapshot and test-set validation.

Planned: snapshot.py fetches the pages in data/corpus/sources.csv (httpx + trafilatura, 1 request/s,
respects robots.txt) into data/corpus/docs/<doc_id>.md; validate.py checks the test set against the
corpus (gold docs exist, every must_include fact appears in a gold doc, duplicates, balance).
Install the scraping extras with `uv sync --group scrape`.
"""
