#!/usr/bin/env python
import argparse
from utils.curator import Curator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="curate molecules")
    parser.add_argument("-i", "--input", help="input files (.smi, .csv, .xlsx)", required=True)
    parser.add_argument("-o", "--output", help="output files(.smi, .csv)", required=True)
    parser.add_argument("-u", "--uncharge", help="neutralize molecule", action="store_true")
    parser.add_argument("-a", "--augment", help="augment SMILES", type=int, default=1)
    parser.add_argument("-p", "--pH_range", nargs="+", type=float)
    args = parser.parse_args()

    if args.uncharge and args.pH_range:
        raise Exception('Arguments uncharge and pH_range cannot be simultaneously used!')
    
    preper = Curator(args.uncharge, args.augment, args.pH_range)
    preper.load(args.input)
    preper.curate_mols()
    preper.save(args.output)