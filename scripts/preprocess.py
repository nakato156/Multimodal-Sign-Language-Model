from src.data_proc.prepr import Preprocess
import argparse
import os

"""
Script para preprocesar el consolidado del dataset.
- Limpia duraciones (> p95)
- Genera un muestreo de los clips limpios o el consolidado completo.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/raw/datasetv1.hdf5")
    parser.add_argument("--output_path", required=False)
    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--fix", action="store_true") # por si se necesita
    args = parser.parse_args()


    base = os.path.splitext(os.path.basename(args.input_path))[0] 
    
    # Determina si el nombre para el sample
    if args.output_path:
        output = args.output_path
    else:
        s = "_preprocessed" if args.sample_rate == 100 else f"_sample{args.sample_rate}_preprocessed"
        output = f"data/processed/{base}{s}.hdf5"
        

    pp = Preprocess(args.input_path, output, seed=42)
    pp.run(sample_rate=args.sample_rate, replace=args.fix)



