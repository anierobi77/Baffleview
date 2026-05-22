import os
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.engine_sr import run_super_res_engine, evaluate_facies_with_benchmarks

def process_single_well(file_path, benchmark_config):
    """
    Worker function executed in parallel across CPU cores.
    Ingests, cleans, and runs the AI pipeline for a single asset log.
    """
    try:
        # Load raw asset data
        df = pd.read_csv(file_path)
        uwi = os.path.basename(file_path).replace(".csv", "")
        
        # Standard structural mapping fallback if headers are messy
        df.columns = [col.upper() for col in df.columns]
        rename_dict = {'DEPTH': 'Depth', 'DEPT': 'Depth', 'GR': 'GR', 'RES': 'Resistivity', 'ILD': 'Resistivity', 'POR': 'Porosity', 'NPHI': 'Porosity'}
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        
        # In-memory evaluation loop
        df_sr = run_super_res_engine(df)
        df_sr['AI_Facies'] = df_sr.apply(lambda row: evaluate_facies_with_benchmarks(
            row['GR_SR'], row['Res_SR'], row['Por_SR'], benchmark_config
        ), axis=1)
        
        # Extract structural metrics
        total_intervals = len(df_sr)
        sh1_baffle_count = len(df_sr[df_sr['AI_Facies'].str.contains("SH1")])
        baffle_density = round((sh1_baffle_count / total_intervals) * 100, 2)
        
        return {
            "UWI": uwi,
            "Status": "SUCCESS",
            "Total_Intervals_Mapped": total_intervals,
            "SH1_Baffle_Count": sh1_baffle_count,
            "Baffle_Density_Pct": baffle_density,
            "Reservoir_Risk": "HIGH" if baffle_density > 15.0 else "LOW"
        }
    except Exception as e:
        return {"UWI": os.path.basename(file_path), "Status": f"FAILED: {str(e)}"}

def run_field_batch_processing(folder_path, benchmark_config):
    """Orchestrates multi-threaded execution across thousands of logs."""
    search_path = os.path.join(folder_path, "*.csv")
    well_files = glob.glob(search_path)
    
    if not well_files:
        print(f"? No matching CSV logs found in target folder: {folder_path}")
        return None
        
    print(f"?? Initializing Batch Processing Core for {len(well_files)} assets...")
    
    batch_summary_results = []
    # Use ProcessPoolExecutor to split files cleanly across available hardware cores
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_well, file, benchmark_config) for file in well_files]
        for idx, future in enumerate(futures):
            res = future.result()
            batch_summary_results.append(res)
            print(f"?? [{idx+1}/{len(well_files)}] Processed: {res['UWI']} -> Status: {res['Status']}")
            
    summary_df = pd.DataFrame(batch_summary_results)
    return summary_df
