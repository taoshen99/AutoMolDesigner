from tqdm import tqdm
import os
import pandas as pd
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit import RDLogger

from molvs import Standardizer
from molvs.fragment import LargestFragmentChooser
from molvs.charge import Uncharger
from dimorphite_dl import DimorphiteDL
from utils.SmilesEnumerator import SmilesEnumerator
RDLogger.DisableLog('rdApp.*')

class Curator():
    def __init__(self, charge_state: bool, aug_fold: int, pH_range=None):
        self.standardizer = Standardizer()
        self.chooser = LargestFragmentChooser()
        if charge_state:
            self.uncharger = Uncharger()
        else:
            self.uncharger = None
        if pH_range:
            self.dimorphite_dl = DimorphiteDL(

                min_ph=pH_range[0],
                max_ph=pH_range[1],
                max_variants=128,
                label_states=False,
                pka_precision=1.0
            )
        else:
            self.dimorphite_dl = None
        self.sme = SmilesEnumerator()
        self.aug_fold = aug_fold
        self.curated_smis = []
        self.curated_auged_smis = []
        self.dict_smi_label_curated = {}

    def load(self, input_data):
        file_type = os.path.splitext(input_data)[1]
        if file_type == ".smi":
            with open(input_data, "r") as f:
                self.smis = f.read().splitlines()
            print(f"loading finished, {len(self.smis)} molecules for curating")
        elif file_type == ".csv":
            df_input_data = pd.read_csv(input_data)
            self.dict_smi_label = {}
            for _, row in df_input_data.iterrows():
                self.dict_smi_label[row[0]] = row[1]
            print(f"loading finished, {len(self.dict_smi_label)} molecules for curating")
        else:
            df_input_data = pd.read_excel(input_data)
            self.dict_smi_label = {}
            for _, row in df_input_data.iterrows():
                self.dict_smi_label[row[0]] = row[1]
            print(f"loading finished, {len(self.dict_smi_label)} molecules for curating")
    
    def save(self, output_data):
        print("saving molecules ...")
        file_type = os.path.splitext(output_data)[1]
        if file_type == ".smi":
            with open(output_data, "w") as f:
                for smi in self.curated_auged_smis:
                    f.write(smi + "\n")
        elif file_type == ".csv":
            df_tabular = pd.DataFrame(self.dict_smi_label_curated.items(), columns=['SMILES', 'label'])
            df_tabular.to_csv(output_data, index=False)
        else:
            df_tabular = pd.DataFrame(self.dict_smi_label_curated.items(), columns=['SMILES', 'label'])
            df_tabular.to_excel(excel_writer=output_data, index=False)
        print(f"saving finished, output {len(self.curated_smis)}×{self.aug_fold} molecules")
    
    def curate(self, mol):
        mol = self.standardizer.standardize(mol)
        mol = self.chooser.choose(mol)
        if self.uncharger:
            mol = self.uncharger.uncharge(mol)
        return mol
    
    def curate_mols(self):
        if hasattr(self, 'smis'):
            for smi in tqdm(self.smis, ascii=True, desc=f"curating molecules"):
                mol = MolFromSmiles(smi)
                if mol:
                    mol = self.curate(mol)
                    if mol:
                        smi = MolToSmiles(mol, isomericSmiles=False)
                        if MolFromSmiles(smi):
                            if smi not in self.curated_smis:
                                if self.dimorphite_dl:
                                    smis_proted = self.dimorphite_dl.protonate(smi)
                                    self.curated_smis.extend(smis_proted)
                                    if self.aug_fold == 1:
                                        self.curated_auged_smis.extend(smis_proted)
                                    else:
                                        for smi_proted in smis_proted:
                                            smis_auged = [self.sme.randomize_smiles(smi_proted) for i in range(self.aug_fold)]
                                            self.curated_auged_smis.extend(smis_auged)                      
                                else:
                                    self.curated_smis.append(smi)
                                    if self.aug_fold == 1:
                                        smis_auged = [smi]
                                    else:
                                        smis_auged = [self.sme.randomize_smiles(smi) for i in range(self.aug_fold)]
                                    self.curated_auged_smis.extend(smis_auged)
        else:
            for smi, label in tqdm(self.dict_smi_label.items(), ascii=True, desc=f"curating molecules"):
                mol = MolFromSmiles(smi)
                if mol:
                    mol = self.curate(mol)
                    if mol:
                        smi = MolToSmiles(mol, isomericSmiles=False)
                        if MolFromSmiles(smi):
                            if smi not in self.curated_smis:
                                if self.dimorphite_dl:
                                    smis_proted = self.dimorphite_dl.protonate(smi)
                                    self.curated_smis.extend(smis_proted)
                                    if self.aug_fold == 1:
                                        for smi_proted in smis_proted:
                                            self.dict_smi_label_curated[smi_proted] = label
                                    else:
                                        for smi_proted in smis_proted:
                                            smis_auged = [self.sme.randomize_smiles(smi_proted) for i in range(self.aug_fold)]
                                            for smi_auged in smis_auged:
                                                self.dict_smi_label_curated[smi_auged] = label
                                else:
                                    self.curated_smis.append(smi)
                                    if self.aug_fold == 1:
                                        self.dict_smi_label_curated[smi] = label
                                    else:
                                        smis_auged = [self.sme.randomize_smiles(smi) for i in range(self.aug_fold)]
                                        for smi_auged in smis_auged:
                                            self.dict_smi_label_curated[smi_auged] = label