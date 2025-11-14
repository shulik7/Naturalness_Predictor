#!/usr/bin/env python3
"""
Script to mark natural molecules in ChEMBL dataset.
"""

import pandas as pd
import sys
from pathlib import Path

def load_natural_molecules(npatlas_file):
    """
    Load natural molecules from NPAtlas dataset.
    
    Args:
        npatlas_file: Path to NPAtlas TSV file
        
    Returns:
        set: Set of inchi keys for natural molecules
    """
    print(f"Loading natural molecules from {npatlas_file}...")
    
    try:
        # Read NPAtlas data
        npatlas_df = pd.read_csv(npatlas_file, sep='\t', low_memory=False)
        
        # Extract inchi keys and remove any NaN values
        natural_inchi_keys = set(npatlas_df['compound_inchikey'].dropna().astype(str).str.upper())

        print(f"Loaded {len(natural_inchi_keys)} unique natural molecule SMILES")
        return natural_inchi_keys
        
    except Exception as e:
        print(f"Error loading NPAtlas data: {e}")
        sys.exit(1)

def mark_chembl_molecules(chembl_file, natural_inchi_keys, output_file):
    """
    Make whehter ChEMBL molecules are natural molecules.
    
    Args:
        chembl_file: Path to ChEMBL TSV file
        natural_inchi_keys: Set of natural molecule Inchi Keys to check
        output_file: Path for output file
    """
    print(f"Loading ChEMBL molecules from {chembl_file}...")
    
    try:
        # Read ChEMBL data in chunks to handle large file size
        chunk_size = 10000
        total_processed = 0
        total_natural = 0
        total_non_natural = 0
        write_header = True

        with pd.io.common.get_handle(output_file, 'w', encoding='utf-8') as out_handle:
            for chunk in pd.read_csv(chembl_file, sep='\t', chunksize=chunk_size, low_memory=False):
                total_processed += len(chunk)
                
                # Remove rows with NaN Inchi Key or SMILES
                chunk = chunk.dropna(subset=['Smiles', 'Inchi Key'])
                
                # Mark out natural molecules
                chunk['Is_Nature_Product'] = chunk['Inchi Key'].str.upper().apply(lambda x: 1 if x in natural_inchi_keys else 0)
            
                # Write the chunk to the output file
                chunk.to_csv(out_handle.handle, sep='\t', columns=['Smiles', 'Is_Nature_Product'], index=False, header=write_header)
                
                total_natural += chunk['Is_Nature_Product'].sum()
                total_non_natural += len(chunk) - total_natural
                
                if total_processed % 50000 == 0:
                    print(f"Processed {total_processed:,} molecules, found {total_natural:,} natural molecules so far...")
                
                # After the first chunk, ensure subsequent chunks don't write the header
                write_header = None
        
        print(f"Final results:")
        print(f"  Total ChEMBL molecules processed: {total_processed:,}")
        print(f"  Natural molecules found: {total_natural:,}")
        print(f"  Non-natural molecules remaining: {total_non_natural:,}")
              
        print("Processing complete!")
        return
        
    except Exception as e:
        print(f"Error processing ChEMBL data: {e}")
        sys.exit(1)

def main():
    """Main function to orchestrate the process."""
    
    # File paths
    script_dir = Path(__file__).parent
    npatlas_file = script_dir / "NPAtlas_download_2024_09.tsv"
    chembl_file = script_dir / "ChEMBL_small_molecule.tsv"
    output_file = script_dir / "marked_ChEMBL_small_molecules.tsv"

    # Check if input files exist
    if not npatlas_file.exists():
        print(f"Error: NPAtlas file not found: {npatlas_file}")
        sys.exit(1)
    
    if not chembl_file.exists():
        print(f"Error: ChEMBL file not found: {chembl_file}")
        sys.exit(1)
    
    print("Starting natural molecule marking process...")
    print("=" * 60)
    
    # Step 1: Load natural molecules
    natural_smiles = load_natural_molecules(npatlas_file)
    
    print("=" * 60)
    
    # Step 2: Mark ChEMBL molecules
    mark_chembl_molecules(chembl_file, natural_smiles, output_file)
    
if __name__ == "__main__":
    main()
