from rdkit.Chem import SanitizeMol, MolToSmiles, MolFromSmiles

class Benchmark:
    def __init__(self, input_smis_path):
        self.smis = self.load_smis(input_smis_path)
        self.valid_smis = self.check_validity()
        self.smis_deduplicated = self.check_uniqueness()

    def load_smis(self, input_smis_path):
        with open(input_smis_path, 'r') as f:
            smis = [smi.rstrip() for smi in f]
        return smis

    def check_validity(self):
        valid_mols = []
        valid_smis = []
        for smi in self.smis:
            mol = MolFromSmiles(smi) 
            if mol:
                try:
                    SanitizeMol(mol)
                except ValueError:
                    print(f"{smi} cannot be sanitized!")
                else:
                    valid_mols.append(mol)
                    valid_smis.append(MolToSmiles(mol))
        self.validity = len(valid_smis) / len(self.smis)
        print(f'Validity: {self.validity:.2%}')
        return valid_smis

    def check_uniqueness(self):
        smis_deduplicated = list(set(self.valid_smis))
        smis_deduplicated.sort(key=self.valid_smis.index)
        self.uniqueness = len(smis_deduplicated) / len(self.valid_smis)
        print(f'Uniqueness: {self.uniqueness:.2%}')
        return smis_deduplicated
    
    def check_novelty(self, known_smis_path):
        """Check novelty of sampled SMILES.

            Args:

                known_smis_path: The known SMILES file which will be compared with the input SMILES file. It should be prepared before use.
        """
        known_smis = self.load_smis(known_smis_path)
        known_smis_cano = []
        for smi in known_smis:
            mol = MolFromSmiles(smi)
            smi_cano = MolToSmiles(mol)
            known_smis_cano.append(smi_cano)
        res_smis = list(set(self.smis_deduplicated) - set(known_smis_cano))
        self.novelty = len(res_smis) / len(self.smis_deduplicated)
        print(f'Novelty: {self.novelty:.2%}')
        res_smis.sort(key=self.smis_deduplicated.index)
        return res_smis
