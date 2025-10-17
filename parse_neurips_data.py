#!/usr/bin/env python3
"""
NeurIPS 2025 Accepted Papers Data Parser

This script loads and parses the NeurIPS 2025 accepted papers JSON file,
providing various analysis and exploration capabilities.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import re
from rich import print


def clean_institution(institution: str) -> str:
    """
    Clean the institution name.
    """
    institution = institution.strip()
    clean_map = {
        "Massachusetts Institute of Technology": "MIT",
        "Stanford University": "Stanford",
        "University of Oxford": "Oxford",
        "Princeton University": "NYU",
        "University of California, Berkeley": "UCB",
        "New York University": "NYU",
        "Tsinghua University": "Tsinghua",
        "Fudan University": "Fudan",
        "Tsinghua University, Tsinghua University": "Tsinghua",
        "Peking University": "PKU",
        "National University of Singapore": "NUS",
        "Nanyang Technological University": "NTU",
        "Shanghai Jiao Tong University": "SJTU",
        "Shanghai Jiaotong University": "SJTU",
        "Carnegie Mellon University": "CMU",
        "University of Science and Technology of China": "USTC",
        "The Chinese University of Hong Kong": "CUHK",
        "The Hong Kong University of Science and Technology": "HKUST",
    }
    if institution in clean_map:
        return clean_map[institution]
    return institution


class NeurIPSPaper:
    """A class representing a NeurIPS paper."""

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the paper.
        """
        self.paper_data = data
        self.id = data.get("id")
        self.uid = data.get("uid")
        self.title = data.get("name")
        self.track = data.get("track")
        self.abstract = data.get("abstract")
        self.decision = data.get("decision")
        self.event_type = data.get("event_type")
        self.session = data.get("session")
        self.topic = data.get("topic")
        self.keywords = data.get("keywords")
        self.num_authors = len(data.get("authors", []))
        self.author_names = [
            author.get("fullname") for author in data.get("authors", [])
        ]
        self.author_institutions = [
            author.get("institution") for author in data.get("authors", [])
        ]
        self.openreview_url = data.get("openreview_url")
        self.paper_url = data.get("paper_url")
        self.virtual_site_url = data.get("virtualsite_url")

    def set_data(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class NeurIPSDataParser:
    """Parser for NeurIPS 2025 accepted papers data."""

    def __init__(self, json_file_path: str):
        """
        Initialize the parser with the JSON file path.

        Args:
            json_file_path: Path to the NeurIPS 2025 JSON file
        """
        self.json_file_path = Path(json_file_path)
        self.data = None
        self.papers_df = None

    def load_data(self) -> Dict[str, Any]:
        """
        Load the JSON data from file.

        Returns:
            Dictionary containing the loaded JSON data
        """
        print(f"Loading data from '{self.json_file_path}'...")

        try:
            with open(self.json_file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            print(
                f"Successfully loaded data with {len(self.data.get('results', []))} papers"
            )
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def parse_papers(self) -> pd.DataFrame:
        """
        Parse the papers data into a pandas DataFrame for easier analysis.

        Returns:
            DataFrame containing paper information
        """
        if self.data is None:
            self.load_data()

        papers = self.data.get("results", [])
        if not papers:
            print("No papers found in the data")
            return pd.DataFrame()

        # Extract key information from each paper
        parsed_papers = dict()
        for paper in papers:

            paper_url = paper.get("paper_url", "")
            sourceurl = paper.get("sourceurl", "")
            virtualsite_url = paper.get("virtualsite_url", "")
            if not paper_url and not virtualsite_url and not sourceurl:
                continue

            if "id=NeurIPS.cc/2025/Conference" in sourceurl:
                track = "NeurIPS 2025 (Main Track)"
            elif "id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track" in sourceurl:
                track = "NeurIPS 2025 (DB Track)"
            elif "id=NeurIPS.cc/2025/Position_Paper_Track" in sourceurl:
                track = "NeurIPS 2025 (Position Paper Track)"
            elif "JmlrOrg" in sourceurl:
                track = "JMLR"
                continue
            elif "annals-of-statistics" in sourceurl:
                track = "Annals of Statistics"
                continue
            else:
                continue

            # Extract authors information
            # for each author, it is a dictionary with keys:
            #     - id
            #     - fullname
            #     - url
            #     - institution
            authors = paper.get("authors", [])
            author_names = [author.get("fullname") for author in authors]
            author_names = list(set([x for x in author_names if x]))
            author_institutions = [author.get("institution") for author in authors]
            author_institutions = list(
                set([clean_institution(x) for x in author_institutions if x])
            )

            # Extract event media (usually OpenReview links)
            event_media = paper.get("eventmedia", [])
            openreview_url = None
            for media in event_media:
                if media.get("name") == "OpenReview":
                    openreview_url = media.get("uri")
                    break

            decision = paper.get("decision", "")
            decision = f"{track} {decision}"
            uid = paper.get("uid")

            parsed_paper = {
                "id": paper.get("id"),
                "uid": paper.get("uid"),
                "title": paper.get("name", ""),
                "track": track,
                "abstract": paper.get("abstract", ""),
                "decision": decision,
                "event_type": paper.get("event_type", ""),
                "session": paper.get("session", ""),
                "topic": paper.get("topic", ""),
                "keywords": paper.get("keywords", []),
                "num_authors": len(authors),
                "author_names": author_names,
                "author_institutions": author_institutions,
                "openreview_url": openreview_url,
                "paper_url": paper.get("paper_url", ""),
                "virtual_site_url": paper.get("virtualsite_url", ""),
                # "start_time": paper.get("starttime"),
                # "end_time": paper.get("endtime"),
                # "room_name": paper.get("room_name"),
                # "latitude": paper.get("latitude"),
                # "longitude": paper.get("longitude"),
            }
            if uid in parsed_papers:
                # print(f"Duplicate UID found: {uid}; Checking DIFFERENCES...")

                # print(f"Parsed paper: {parsed_papers[uid]}")
                # print(f"New parsed paper: {parsed_paper}")
                title_diff = False
                title_i = parsed_papers[uid]["title"]
                title_j = parsed_paper["title"]
                if title_i != title_j:
                    title_diff = True

                if title_diff:
                    new_uid = uid + "_diff"
                    parsed_paper["uid"] = new_uid
                    parsed_papers[new_uid] = parsed_paper
                    # print("=" * 60)
                    # print("[red bold]Different titles with the same UID!")

                    # print(f"Title 1: {title_i}")
                    # print(f"Title 2: {title_j}")
                    # print(f"track 1: {parsed_papers[uid]['track']}")
                    # print(f"track 2: {parsed_paper['track']}")
                    # print("=" * 60)
                else:
                    # print("Duplicated paper entry found!")
                    parsed_papers[uid]["new_id"] = parsed_paper["id"]
                    parsed_papers[uid]["new_openreview_url"] = parsed_paper[
                        "openreview_url"
                    ]
                    parsed_papers[uid]["new_virtual_site_url"] = parsed_paper[
                        "virtual_site_url"
                    ]
                    # for _key in parsed_papers[uid]:
                    #     if _key not in parsed_paper:
                    #         print(f"Key {_key} not in new parsed paper")
                    #         continue
                    #     if parsed_papers[uid][_key] != parsed_paper[_key]:
                    #         print(
                    #             f"Value {parsed_papers[uid][_key]} != {parsed_paper[_key]} for key {_key}"
                    #         )

            else:
                parsed_papers[uid] = parsed_paper

        self.papers_df = pd.DataFrame(list(parsed_papers.values()))
        # print(f"Parsed {len(self.papers_df)} papers into DataFrame")
        return self.papers_df

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the papers.

        Returns:
            Dictionary containing summary statistics
        """
        if self.papers_df is None:
            self.parse_papers()

        stats = {
            "total_papers": len(self.papers_df),
            "total_count_from_api": self.data.get("count", 0),
            "decisions": dict(self.papers_df["decision"].value_counts()),
            "event_types": dict(self.papers_df["event_type"].value_counts()),
            "papers_with_topics": len(
                self.papers_df[
                    self.papers_df["topic"].notna() & (self.papers_df["topic"] != "")
                ]
            ),
            "papers_with_keywords": len(
                self.papers_df[self.papers_df["keywords"].apply(lambda x: len(x) > 0)]
            ),
            "papers_with_openreview": len(
                self.papers_df[self.papers_df["openreview_url"].notna()]
            ),
            "avg_authors_per_paper": self.papers_df["num_authors"].mean(),
            "max_authors": self.papers_df["num_authors"].max(),
            "min_authors": self.papers_df["num_authors"].min(),
        }

        return stats

    def get_top_institutions(self, top_n: int = 20) -> List[tuple]:
        """
        Get the top institutions by number of papers.

        Args:
            top_n: Number of top institutions to return

        Returns:
            List of tuples (institution, count)
        """
        if self.papers_df is None:
            self.parse_papers()

        # Flatten all institutions
        all_institutions = []
        for institutions in self.papers_df["author_institutions"]:
            all_institutions.extend(institutions)

        # Count institutions
        institution_counts = Counter(all_institutions)
        return institution_counts.most_common(top_n)

    def get_top_authors(self, top_n: int = 20) -> List[tuple]:
        """
        Get the top authors by number of papers.

        Args:
            top_n: Number of top authors to return

        Returns:
            List of tuples (author, count)
        """
        if self.papers_df is None:
            self.parse_papers()

        # Flatten all authors
        all_authors = []
        for authors in self.papers_df["author_names"]:
            all_authors.extend(authors)

        # Count authors
        author_counts = Counter(all_authors)
        return author_counts.most_common(top_n)

    def search_papers(
        self, query: str, search_fields: List[str] = None
    ) -> pd.DataFrame:
        """
        Search papers by query string.

        Args:
            query: Search query
            search_fields: Fields to search in (default: title, abstract, author_names)

        Returns:
            DataFrame with matching papers
        """
        if self.papers_df is None:
            self.parse_papers()

        if search_fields is None:
            search_fields = ["title", "abstract", "author_names"]

        query_lower = query.lower()
        mask = pd.Series([False] * len(self.papers_df))

        for field in search_fields:
            if field == "author_names":
                # Special handling for author names
                author_mask = self.papers_df[field].apply(
                    lambda authors: any(
                        query_lower in author.lower() for author in authors
                    )
                )
                mask = mask | author_mask
            else:
                field_mask = (
                    self.papers_df[field]
                    .str.lower()
                    .str.contains(query_lower, na=False)
                )
                mask = mask | field_mask

        return self.papers_df[mask].copy()

    def filter_by_decision(self, decision: str) -> pd.DataFrame:
        """
        Filter papers by decision type.

        Args:
            decision: Decision type (e.g., 'Accept (poster)', 'Accept (oral)')

        Returns:
            DataFrame with filtered papers
        """
        if self.papers_df is None:
            self.parse_papers()

        return self.papers_df[self.papers_df["decision"] == decision].copy()

    def filter_by_track(self, track: str) -> pd.DataFrame:
        """
        Filter papers by track.

        Args:
            track: Track to filter by

        Returns:
            DataFrame with filtered papers
        """
        if self.papers_df is None:
            self.parse_papers()

        return self.papers_df[self.papers_df["track"] == track].copy()

    def filter_by_topic(self, topic: str) -> pd.DataFrame:
        """
        Filter papers by topic.

        Args:
            topic: Topic to filter by

        Returns:
            DataFrame with filtered papers
        """
        if self.papers_df is None:
            self.parse_papers()

        return self.papers_df[
            self.papers_df["topic"].str.contains(topic, case=False, na=False)
        ].copy()

    def get_papers_by_institution(self, institution: str) -> pd.DataFrame:
        """
        Get papers by institution.

        Args:
            institution: Institution name to search for

        Returns:
            DataFrame with papers from the institution
        """
        if self.papers_df is None:
            self.parse_papers()

        institution_lower = institution.lower()
        mask = self.papers_df["author_institutions"].apply(
            lambda institutions: any(
                institution_lower in inst.lower() for inst in institutions
            )
        )

        return self.papers_df[mask].copy()

    def export_to_csv(self, output_path: str, filtered_df: pd.DataFrame = None):
        """
        Export papers data to CSV.

        Args:
            output_path: Output CSV file path
            filtered_df: Optional filtered DataFrame to export
        """
        if filtered_df is not None:
            df_to_export = filtered_df
        elif self.papers_df is not None:
            df_to_export = self.papers_df
        else:
            self.parse_papers()
            df_to_export = self.papers_df

        # Convert list columns to strings for CSV export
        df_export = df_to_export.copy()
        df_export["author_names"] = df_export["author_names"].apply(
            lambda x: "; ".join(x)
        )
        df_export["author_institutions"] = df_export["author_institutions"].apply(
            lambda x: "; ".join(x)
        )
        df_export["keywords"] = df_export["keywords"].apply(lambda x: "; ".join(x))

        df_export.to_csv(output_path, index=False)
        print(f"\nExported {len(df_export)} papers to {output_path}")

    def print_summary(self):
        """Print a summary of the data."""
        stats = self.get_summary_stats()

        print("\n" + "=" * 60)
        print("NEURIPS 2025 ACCEPTED PAPERS SUMMARY")
        print("=" * 60)
        # print(f"Total count from API: {stats['total_count_from_api']}")
        print(f"Total NeurIPS 2025 accepted papers: {stats['total_papers']}")
        print(f"Average authors per paper: {stats['avg_authors_per_paper']:.1f}")
        # print(f"Papers with topics: {stats['papers_with_topics']}")
        # print(f"Papers with keywords: {stats['papers_with_keywords']}")
        # print(f"Papers with OpenReview links: {stats['papers_with_openreview']}")

        print("\nDECISIONS:")
        decision_keys = sorted(list(stats["decisions"].keys()))
        for decision in decision_keys:
            print(f"  {decision}: {stats['decisions'][decision]}")


def main():
    """Main function to run the parser with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parse NeurIPS 2025 accepted papers data"
    )
    parser.add_argument(
        "--data-path",
        help="Path to the NeurIPS JSON file",
        type=str,
        default="neurips-2025-orals-posters.json",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary statistics"
    )
    parser.add_argument("--search", type=str, help="Search query for papers")
    parser.add_argument("--institution", type=str, help="Filter by institution")
    parser.add_argument("--decision", type=str, help="Filter by decision type")
    parser.add_argument("--track", type=str, help="Filter by track")
    parser.add_argument("--topic", type=str, help="Filter by topic")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV file")
    parser.add_argument(
        "--top-institutions",
        type=int,
        default=0,
        help="Number of top institutions to show",
    )
    parser.add_argument(
        "--top-authors", type=int, default=0, help="Number of top authors to show"
    )

    args = parser.parse_args()

    # Initialize parser
    neurips_parser = NeurIPSDataParser(args.data_path)

    # Load and parse data
    neurips_parser.load_data()
    neurips_parser.parse_papers()

    # Apply filters
    filtered_df = neurips_parser.papers_df.copy()

    if args.institution:
        filtered_df = neurips_parser.get_papers_by_institution(args.institution)
        print(f"\nFound {len(filtered_df)} papers from {args.institution}")

    if args.track:
        filtered_df = neurips_parser.filter_by_track(args.track)
        print(f"\nFound {len(filtered_df)} papers from track: {args.track}")

    if args.decision:
        filtered_df = neurips_parser.filter_by_decision(args.decision)
        print(f"\nFound {len(filtered_df)} papers with decision: {args.decision}")

    if args.topic:
        filtered_df = neurips_parser.filter_by_topic(args.topic)
        print(f"\nFound {len(filtered_df)} papers with topic containing: {args.topic}")

    if args.search:
        filtered_df = neurips_parser.search_papers(args.search)
        print(f"\nFound {len(filtered_df)} papers matching search: '{args.search}'")

    # Print summary if requested
    if args.summary:
        neurips_parser.print_summary()

    # Show top institutions and authors
    if args.top_institutions > 0:
        print(f"\nTOP {args.top_institutions} INSTITUTIONS:")
        top_institutions = neurips_parser.get_top_institutions(args.top_institutions)
        for i, (institution, count) in enumerate(top_institutions, 1):
            print(f"  {i:2d}. {institution}: {count} papers")

    if args.top_authors > 0:
        print(f"\nTOP {args.top_authors} AUTHORS:")
        top_authors = neurips_parser.get_top_authors(args.top_authors)
        for i, (author, count) in enumerate(top_authors, 1):
            print(f"  {i:2d}. {author}: {count} papers")

    # Export to CSV if requested
    if args.export_csv:
        neurips_parser.export_to_csv(args.export_csv, filtered_df)


if __name__ == "__main__":
    main()
