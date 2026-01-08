# Forensic Signatures of Bandwidth-Limited Quantum Control

Analysis pipeline for detecting hesitation signatures in quantum systems and gravitational wave detectors.

## Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/aernouddekker/forensic-signatures.git
cd forensic-signatures

# 2. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# 3. Run the complete pipeline
cd scripts
python run_all.py --clean --n_events 500  # ~30 GB download, ~30 min
```

## Pipeline Options

```bash
python run_all.py --clean --n_events 10   # Quick test (~1 GB, ~5 min)
python run_all.py --clean --n_events 500  # Recommended (~30 GB, ~30 min)
python run_all.py --cached                # Use cached data (~5 min)
python run_all.py --macros                # Regenerate macros/PDF only
python run_all.py                         # Full catalog (~168 GB)
```

## Data Sources

The pipeline automatically downloads all required datasets:

- **LIGO strain data**: Gravity Spy O3a catalog from GWOSC
- **26-qubit data**: McEwen et al. (2022) from Figshare
- **63-qubit data**: Li et al. (2025) from Figshare

## Output

After successful completion:
- `latex/signatures_submission.pdf` - Complete manuscript with all figures
- `scripts/output/` - JSON results from all analyses
- `figures/` - Generated figures

## Requirements

- Python 3.9+
- ~30-170 GB disk space (depending on LIGO event count)
- Network access for initial data download

## Citation

See the manuscript for full citation information.

## License

MIT License - see LICENSE file.

## Links

- OSF Project: https://doi.org/10.17605/OSF.IO/Q8TV2
- Main IOF Framework: https://ignorantobserver.xyz
