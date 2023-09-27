from utils.benchmark import Benchmark
import argparse

def main(input_smis_path, output_smis_path, known_smis_path):
    benchmark_clm = Benchmark(input_smis_path)

    if known_smis_path is None:
        with open(output_smis_path, 'w') as f:
            for smi in benchmark_clm.smis_deduplicated:
                f.write(smi + '\n')
        print('Validity and uniqueness have been checked.')
        print(f'SMILESs have been output to {output_smis_path}.')
    else:
        res_smi = benchmark_clm.check_novelty(known_smis_path)
        with open(output_smis_path, 'w') as f:
            for smi in res_smi:
                f.write(smi + '\n')
        print('Validity, uniqueness and novelty have been checked.')
        print(f'SMILESs have been output to {output_smis_path}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark sampled SMILES")
    parser.add_argument("-i", "--input", required=True, help="input file")
    parser.add_argument("-o", "--output", default="smiles_curated.smi", help="output file")
    parser.add_argument("-k", "--known", default=None, help="known SMILES")
    args = parser.parse_args()

    main(args.input, args.output, args.known)