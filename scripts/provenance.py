#!/usr/bin/env python3
"""
Provenance Module
=================

Provides functions to add provenance metadata to JSON outputs for reproducibility.
Each JSON output should include a 'provenance' block with:
- script name and version
- git commit hash (if available)
- timestamp
- input data sources (DOIs, filenames, hashes)
- parameter configuration

Usage:
    from provenance import get_provenance, add_provenance, hash_file

    # Get provenance for current script
    prov = get_provenance(__file__, params={'threshold': 2, 'window_ms': 100})

    # Add provenance to existing dict
    output = {'results': [...]}
    add_provenance(output, __file__, params={...})

    # Hash an input file
    file_hash = hash_file('/path/to/data.json')

Author: Aernoud Dekker
Date: December 2025
"""

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union

# =============================================================================
# Version
# =============================================================================

__version__ = "1.0.0"

# Known data sources
DATA_SOURCES = {
    'mcewan': {
        'doi': '10.6084/m9.figshare.16673851',
        'description': 'McEwen et al. (2022) cosmic ray telemetry',
        'reference': 'Nature Physics 18, 107-111 (2022)',
    },
    'ligo_gwosc': {
        'doi': '10.7935/K5RN35SH',
        'description': 'LIGO O3a strain data from GWOSC',
        'reference': 'LIGO Scientific Collaboration (2021)',
    },
    'gravity_spy': {
        'doi': '10.5281/zenodo.1476156',
        'description': 'Gravity Spy glitch classifications',
        'reference': 'Zevin et al. (2017)',
    },
}


# =============================================================================
# Git Integration
# =============================================================================

def get_git_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_dirty() -> bool:
    """Check if the working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


# =============================================================================
# File Hashing
# =============================================================================

def hash_file(filepath: Union[str, Path], algorithm: str = 'sha256') -> Optional[str]:
    """
    Compute hash of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('sha256', 'md5', etc.)

    Returns:
        Hex digest string, or None if file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    hasher = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_string(s: str, algorithm: str = 'sha256') -> str:
    """Compute hash of a string."""
    hasher = hashlib.new(algorithm)
    hasher.update(s.encode('utf-8'))
    return hasher.hexdigest()


# =============================================================================
# Provenance Generation
# =============================================================================

def get_provenance(
    script_path: Union[str, Path],
    params: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Union[str, Path]]] = None,
    data_source: Optional[str] = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate provenance metadata for a script run.

    Args:
        script_path: Path to the script (__file__)
        params: Dictionary of analysis parameters
        inputs: Dictionary of input file paths (will be hashed)
        data_source: Key from DATA_SOURCES (e.g., 'mcewan', 'ligo_gwosc')
        rng_seed: Random seed used for reproducibility

    Returns:
        Dictionary with provenance metadata
    """
    script_path = Path(script_path)

    prov = {
        'script': script_path.name,
        'script_version': __version__,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'rng_seed': rng_seed,
    }

    # Git info
    git_hash = get_git_hash()
    if git_hash:
        prov['git_commit'] = git_hash
        if get_git_dirty():
            prov['git_dirty'] = True

    # Parameters
    if params:
        prov['parameters'] = params

    # Input files with hashes
    if inputs:
        prov['inputs'] = {}
        for name, path in inputs.items():
            path = Path(path)
            prov['inputs'][name] = {
                'filename': path.name,
                'path': str(path),
            }
            file_hash = hash_file(path)
            if file_hash:
                prov['inputs'][name]['sha256'] = file_hash

    # Data source info
    if data_source and data_source in DATA_SOURCES:
        prov['data_source'] = DATA_SOURCES[data_source].copy()
        prov['data_source']['key'] = data_source

    return prov


def add_provenance(
    output: Dict[str, Any],
    script_path: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Add provenance metadata to an output dictionary.

    Args:
        output: Dictionary to add provenance to
        script_path: Path to the script (__file__)
        **kwargs: Additional arguments passed to get_provenance()

    Returns:
        The output dictionary with 'provenance' key added
    """
    output['provenance'] = get_provenance(script_path, **kwargs)
    return output


# =============================================================================
# Validation
# =============================================================================

def validate_provenance(data: Dict[str, Any]) -> bool:
    """
    Check if a loaded JSON has valid provenance metadata.

    Returns True if provenance block exists and has required fields.
    """
    if 'provenance' not in data:
        return False

    prov = data['provenance']
    required = ['script', 'timestamp_utc', 'rng_seed']

    return all(field in prov for field in required)


def get_provenance_summary(data: Dict[str, Any]) -> str:
    """
    Get a one-line summary of provenance for display.
    """
    if 'provenance' not in data:
        return "No provenance data"

    prov = data['provenance']
    parts = [prov.get('script', '?')]

    if 'git_commit' in prov:
        commit = prov['git_commit']
        if prov.get('git_dirty'):
            commit += '*'
        parts.append(f"@{commit}")

    if 'timestamp_utc' in prov:
        # Parse and format timestamp
        try:
            ts = datetime.fromisoformat(prov['timestamp_utc'].replace('Z', '+00:00'))
            parts.append(ts.strftime('%Y-%m-%d'))
        except:
            pass

    return ' '.join(parts)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    print("Provenance Module Test")
    print("=" * 50)

    # Test git info
    print(f"\nGit commit: {get_git_hash()}")
    print(f"Git dirty: {get_git_dirty()}")

    # Test provenance generation
    prov = get_provenance(
        __file__,
        params={'threshold': 2, 'window_ms': 100},
        data_source='ligo_gwosc',
    )

    print("\nGenerated provenance:")
    import json
    print(json.dumps(prov, indent=2))
