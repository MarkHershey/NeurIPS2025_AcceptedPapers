#!/usr/bin/env python3
"""
Parser for workshop_raw.html that extracts event data and exports to CSV.
"""

import csv
import re
from bs4 import BeautifulSoup
from pathlib import Path


def clean_text(text):
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace and newlines
    text = re.sub(r"\s+", " ", text.strip())
    return text


def extract_abstract(abstract_div):
    """Extract abstract text from the abstract div."""
    if not abstract_div:
        return ""

    paragraphs = abstract_div.find_all("p")
    if paragraphs:
        # Join all paragraphs with spaces
        abstract_text = " ".join([clean_text(p.get_text()) for p in paragraphs])
        return abstract_text

    # Fallback: get all text
    return clean_text(abstract_div.get_text())


def parse_workshop_html(html_file_path):
    """
    Parse workshop HTML file and extract event data.

    Args:
        html_file_path: Path to the HTML file

    Returns:
        List of dictionaries containing event data
    """
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    events = []
    event_cards = soup.find_all("div", class_="event-card")

    for card in event_cards:
        # Extract event ID
        event_id = card.get("data-event-id", "")

        # Extract event type
        event_type = card.get("data-event-type", "")

        # Extract event title (from data attribute and from h3 text)
        title_slug = card.get("data-event-title", "")
        title_elem = card.find("h3", class_="event-title")
        title_text = ""
        url = ""
        if title_elem:
            link = title_elem.find("a")
            if link:
                title_text = clean_text(link.get_text())
                url = link.get("href", "")

        # Extract speakers
        speakers_elem = card.find("div", class_="event-speakers")
        speakers = clean_text(speakers_elem.get_text()) if speakers_elem else ""

        # Extract time and location
        meta_row = card.find("div", class_="event-meta-row")
        time = ""
        location = ""
        if meta_row:
            time_elem = meta_row.find("span", class_="touchup-time")
            if time_elem:
                time = clean_text(time_elem.get_text())

            # Location is in the second meta-pill span
            meta_pills = meta_row.find_all("span", class_="meta-pill")
            for pill in meta_pills:
                if "time" not in pill.get("class", []):
                    location_span = pill.find("span")
                    if location_span:
                        location = clean_text(location_span.get_text())
                        break

        # Extract abstract
        abstract_elem = card.find("div", class_="abstract-text")
        abstract = extract_abstract(abstract_elem) if abstract_elem else ""

        event_data = {
            "event_id": event_id,
            "event_type": event_type,
            "title": title_text,
            "title_slug": title_slug,
            "url": url,
            "speakers": speakers,
            "time": time,
            "location": location,
            "abstract": abstract,
        }

        events.append(event_data)

    return events


def export_to_csv(events, output_file_path):
    """
    Export events data to CSV file.

    Args:
        events: List of event dictionaries
        output_file_path: Path to output CSV file
    """
    if not events:
        print("No events to export.")
        return

    fieldnames = [
        "event_id",
        "event_type",
        "title",
        "title_slug",
        "url",
        "speakers",
        "time",
        "location",
        "abstract",
    ]

    with open(output_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    print(f"Exported {len(events)} events to {output_file_path}")


def main():
    """Main function to parse HTML and export to CSV."""
    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Input and output file paths
    html_file = script_dir / "workshop_raw.html"
    csv_file = script_dir / "workshop_events.csv"

    if not html_file.exists():
        print(f"Error: {html_file} not found!")
        return

    print(f"Parsing {html_file}...")
    events = parse_workshop_html(html_file)

    print(f"Found {len(events)} events")
    print(f"Exporting to {csv_file}...")
    export_to_csv(events, csv_file)

    print("Done!")


if __name__ == "__main__":
    main()
