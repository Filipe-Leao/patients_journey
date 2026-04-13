import numpy as np
import pandas as pd
import os

def split_and_save_dataset(path, n_splits=4):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, path) 
    case_report = pd.read_csv(file_path)
    
    splits = np.array_split(case_report, n_splits)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    for i, split in enumerate(splits):
        split_path = os.path.join(BASE_DIR, path[:-4] + f"_part{i+1}.csv")
        split.to_csv(split_path, index=False, encoding='utf-8-sig')
        print(f"Parte {i+1} guardada: {len(split)} linhas → {split_path}")
        
split_and_save_dataset("../data/PMCPatients.csv")