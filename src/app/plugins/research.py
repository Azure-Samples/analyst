import os
import csv
import time
import io
import re
import requests

from bs4 import BeautifulSoup, Tag
from typing import Callable, List, Dict

from azure.cognitiveservices.search.websearch import WebSearchClient
from msrest.authentication import CognitiveServicesCredentials
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class ResearchPlugin:

    def __init__(self, bing_subscription_key: str, bing_endpoint: str = "https://api.bing.microsoft.com"):
        self.bing_key = bing_subscription_key
        self.bing_endpoint = bing_endpoint
        self.client = WebSearchClient(
            endpoint=self.bing_endpoint,
            credentials=CognitiveServicesCredentials(self.bing_key)
        )

    @kernel_function(
        name="SearchTopic",
        description="Uses the Bing Web Search SDK to retrieve results for the given topic. Returns a list of results with keys: name, url, and snippet."
    )
    def search_topic(self, topic: str) -> List[dict]:
        search_result = self.client.web.search(query=topic)
        results = []
        if search_result.web_pages and search_result.web_pages.value:  # type: ignore
            for item in search_result.web_pages.value:  # type: ignore
                results.append({
                    "name": item.name,
                    "url": item.url,
                    "snippet": item.snippet
                })
        return results

    @kernel_function(
        name="ConsolidateToCSV",
        description="Consolidates a list of website results into CSV text format. Each row contains: name, url, snippet."
    )
    def consolidate_to_csv(self, results: List[dict]) -> str:
        output = io.StringIO()
        fieldnames = ["name", "url", "snippet"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)
        return output.getvalue()

    @kernel_function(
        name="SimilarSearchLoop",
        description="Continuously searches for similar content for the given topic until the stop condition is met. Returns the CSV text when the stop condition is met."
    )
    def similar_search_loop(self, topic: str, stop_condition: Callable, delay: int = 5) -> str:
        while True:
            results = self.search_topic(topic)
            csv_text = self.consolidate_to_csv(results)
            if stop_condition(csv_text):
                break
            time.sleep(delay)
        return csv_text

    @kernel_function(
        name="ScrapeUrlsForQueryParams",
        description="For each search result URL, scrapes the HTML content and filters out all anchor tags whose href attribute contains a query parameter. Returns a dictionary mapping each URL to a list of filtered href values."
    )
    def scrape_urls_for_query_params(self, results: List[dict]) -> Dict[str, List[str]]:
        filtered_links = {}
        for result in results:
            url = result.get("url", "")
            links_with_query = []
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    if not isinstance(a_tag, Tag):
                        continue
                    href = a_tag.get("href")
                    if href and "?" in href:
                        links_with_query.append(href)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
            filtered_links[url] = links_with_query
        return filtered_links

    @kernel_function(
        name="ExtractTextFromUrls",
        description="Given a list of URLs, retrieves the HTML content and extracts its plain text. Returns a dictionary mapping each URL to its extracted text."
    )
    def extract_text_from_urls(self, urls: List[str]) -> Dict[str, str]:
        extracted_texts = {}
        tag_re = re.compile(r'<[^>]+>')
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                html_content = response.text
                html_content = re.sub(r'<(script|style).*?>.*?</\1>', '', html_content, flags=re.DOTALL)
                text = tag_re.sub('', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
                extracted_texts[url] = text
            except Exception as e:
                print(f"Error extracting text from {url}: {e}")
                extracted_texts[url] = ""
        return extracted_texts

    @kernel_function(
        name="ConsolidateTextsIntoTerms",
        description="Consolidates texts from a list of record dictionaries by classifying their content based on provided topics. For each record, counts the occurrence of each topic in the 'snippet' and adds a 'text_tags' key with the counts. Returns the updated list of dictionaries."
    )
    def consolidate_texts_into_terms(self, records: List[dict], topics: List[str]) -> List[dict]:
        for record in records:
            snippet = record.get("snippet", "")
            tag_counts = {}
            for topic in topics:
                count = snippet.lower().count(topic.lower())
                if count:
                    tag_counts[topic] = count
            record["text_tags"] = tag_counts
        return records
