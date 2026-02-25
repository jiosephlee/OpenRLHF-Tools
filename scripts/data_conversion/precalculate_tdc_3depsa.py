import os
import sys

# Set environment variables to avoid nested parallelism overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import logging
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# Add project root to path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir.parent.parent))

# Mock pyPgSQL to prevent RDKit from crashing when trying to import it
try:
    import pyPgSQL
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["pyPgSQL"] = MagicMock()

from tools.ePSA_3D import get_3d_exposed_polar_surface

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_tdc_smiles():
    smiles_set = set()
    raw_dir = current_dir.parent.parent / 'data' / 'tdc' / 'raw'
    
    if not raw_dir.exists():
        logger.warning(f"Raw data directory {raw_dir} does not exist.")
        return list(smiles_set)
        
    csv_files = list(raw_dir.rglob('*.csv'))
    for csv_file in tqdm(csv_files, desc="Collecting SMILES from local CSVs"):
        try:
            df = pd.read_csv(csv_file)
            if 'Drug' in df.columns:
                smiles_set.update(df['Drug'].dropna().astype(str).unique())
            # For ADME/Tox datasets, they usually have 'Drug' column.
        except Exception as e:
            logger.warning(f"Error processing {csv_file}: {e}")

    return list(smiles_set)

def process_single_smiles(smiles):
    try:
        # Check if valid string
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        
        # Call the ePSA_3D tool function
        desc = get_3d_exposed_polar_surface(smiles)
        return smiles, desc
    except Exception as e:
        return smiles, f"ERROR: {e}"

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


def main():
    output_path = current_dir.parent.parent / 'data' / 'tdc' / 'metadata' / 'TDC_all_3depsa.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect SMILES
    logger.info("Collecting SMILES from TDC...")
    all_smiles = get_all_tdc_smiles()
    logger.info(f"Collected {len(all_smiles)} unique SMILES.")
    
    # 2. Process in parallel
    results = {}
    
    # Check if joblib is available
    try:
        from joblib import Parallel, delayed
        logger.info("Using joblib for parallelism.")
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            parallel = Parallel(n_jobs=64, verbose=5, backend='loky', return_generator=True)
            
            results_generator = parallel(delayed(process_single_smiles)(s) for s in all_smiles)
            
            for res in tqdm(results_generator, total=len(all_smiles), desc="Processing"):
                if res:
                    s, d = res
                    f_out.write(json.dumps({s: d}) + '\n')
                    f_out.flush()
                
    except ImportError:
        logger.info("joblib not found, falling back to multiprocessing.Pool with NoDaemonContext (if nested) or ProcessPoolExecutor.")
        
        class NoDaemonPool(multiprocessing.pool.Pool):
            def Process(self, *args, **kwds):
                proc = super(NoDaemonPool, self).Process(*args, **kwds)
                proc.daemon = False
                return proc
        
        logger.info("Using custom NoDaemonPool.")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            with NoDaemonPool(processes=os.cpu_count()) as pool:
                for res in tqdm(pool.imap_unordered(process_single_smiles, all_smiles), total=len(all_smiles)):
                    if res:
                        s, d = res
                        f_out.write(json.dumps({s: d}) + '\n')
                        f_out.flush()

    # 3. Done
    logger.info(f"Finished processing. Saved to {output_path}")

if __name__ == "__main__":
    main()
