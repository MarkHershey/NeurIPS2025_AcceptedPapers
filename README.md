# NeurIPS 2025 Accepted Papers

This repository contains tools to load, parse, and analyze the NeurIPS 2025 accepted papers data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The parser can be used from the command line with various options:

```bash
# Print summary statistics
python parse_neurips_data.py --summary

# Show top institutions and authors
python parse_neurips_data.py --top-institutions 10 --top-authors 10
```

## License

This project is for educational and research purposes. The data is from the official NeurIPS 2025 conference.
