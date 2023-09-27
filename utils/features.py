import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


def gen_features(m: Chem.Mol, features='ecfp4') -> np.ndarray:
    if features == 'ecfp4':
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        return generator.GetFingerprintAsNumPy(m)
    elif features == 'fcfp6':
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=2048,
                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
        return generator.GetFingerprintAsNumPy(m)
    elif features == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(m)
        fp = np.zeros(len(keys), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(keys, fp)
        return fp
    elif features == 'rdkit_2d':
        smi = Chem.MolToSmiles(m)
        generator = rdDescriptors.RDKit2D()
        return generator.process(smi)[1:] 
    elif features == 'rdkit_2d_norm':
        smi = Chem.MolToSmiles(m)
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        return generator.process(smi)[1:] 
    else:
        raise Exception('Not supported features!')