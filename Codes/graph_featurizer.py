import numpy as np
from typing import Optional, Tuple, List
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Union, Dict, Sequence, Set
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.utils.molecule_feature_utils import one_hot_encode

SMARTS_GROUPS = {
    'carbonyl': '[CX3]=[OX1]',
    'aldehyde': '[CX3H1](=O)[#6]',
    'ketone': '[CX3](=O)[#6]',
    'carboxylic_acid': 'C(=O)[OH]',
    'ester': '[CX3](=O)[OX2H0][#6]',
    'ether': '[OD2]([#6])[#6]',
    'amine_primary': '[NX3;H2;!$(NC=O)]',
    'amine_secondary': '[NX3;H1;!$(NC=O)]',
    'thiol': '[SX2H]',
    'thioether': '[#16X2][#6]',
    'aromatic_ring': 'a1aaaaa1',
    'phenol': 'c1ccc(cc1)[OH]',
    'alkene': 'C=C',
    'alkyne': 'C#C',
    'nitro': '[$([NX3](=O)=O)]'
}
SMARTS_PATTERNS = {name: Chem.MolFromSmarts(s) for name, s in SMARTS_GROUPS.items()}

def get_functional_group_tags(mol: Chem.Mol) -> Dict[str, Set[int]]:
    tag_dict = {name: set() for name in SMARTS_GROUPS}
    for name, pattern in SMARTS_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            tag_dict[name].update(match)
    return tag_dict

class GraphConstants(object):
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES: Dict[str, List[int]] = {
        'valence': [0, 1, 2, 3, 4, 5, 6],
        'degree': [0, 1, 2, 3, 4, 5],
        'num_Hs': [0, 1, 2, 3, 4],
        'formal_charge': [-1, -2, 1, 2, 3, 0],
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'hybridization': ["SP", "SP2", "SP3"]
    }
    ATOM_FDIM = 147
    BOND_FDIM = 7

def atom_features(atom: Chem.Atom, mol: Chem.Mol, atom_idx: int, fg_tags: Dict[str, Set[int]]) -> Sequence[Union[bool, int, float]]:
    if atom is None:
        features: Sequence[Union[bool, int, float]] = [0] * GraphConstants.ATOM_FDIM
    else:
        features = []
        features += one_hot_encode(atom.GetTotalValence(), GraphConstants.ATOM_FEATURES['valence'])
        features += one_hot_encode(atom.GetTotalDegree(), GraphConstants.ATOM_FEATURES['degree'])
        features += one_hot_encode(atom.GetTotalNumHs(), GraphConstants.ATOM_FEATURES['num_Hs'])
        features += one_hot_encode(atom.GetFormalCharge(), GraphConstants.ATOM_FEATURES['formal_charge'])
        features += one_hot_encode(atom.GetAtomicNum()-1, GraphConstants.ATOM_FEATURES['atomic_num'])
        features += [int(atom.GetIsAromatic())]
        features += [int(atom.IsInRing())]
        features += [int(atom.HasProp("_ChiralityPossible"))]
        features += one_hot_encode(str(atom.GetHybridization()), GraphConstants.ATOM_FEATURES['hybridization'])

        try:
            val = atom.GetProp('_GasteigerCharge')
            gasteiger_charge = float(val)
            if not np.isfinite(gasteiger_charge):
                gasteiger_charge = 0.0
        except KeyError:
            gasteiger_charge = 0.0
        features += [gasteiger_charge]
        atomic_mass = atom.GetMass()
        features += [atomic_mass]    
        for name in SMARTS_GROUPS:
            features.append(1.0 if atom_idx in fg_tags[name] else 0.0)
    return features


def bond_features(bond: RDKitBond) -> Sequence[Union[bool, int, float]]:
    if bond is None:
        b_features: Sequence[Union[bool, int, float]] = [1] + [0] * (GraphConstants.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        b_features = [
            0, bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
    return b_features

def compute_mol_descriptors(mol: Chem.Mol) -> np.ndarray:
    desc = []

    desc.append(rdMolDescriptors.CalcExactMolWt(mol))
    desc.append(rdMolDescriptors.CalcNumHeavyAtoms(mol))
    num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    desc.append(num_aromatic_atoms)
    desc.append(Descriptors.TPSA(mol))
    desc.append(Descriptors.NumHDonors(mol))
    desc.append(Descriptors.NumHAcceptors(mol))
    desc.append(Descriptors.MolLogP(mol))
    desc.append(Descriptors.NumRotatableBonds(mol))
    desc.append(Descriptors.RingCount(mol))
    desc.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    desc.append(rdMolDescriptors.CalcFractionCSP3(mol))
    desc.append(Chem.GetFormalCharge(mol))
    try:
        AllChem.ComputeGasteigerCharges(mol)
        g_charges = []
        for atom in mol.GetAtoms():
            try:
                val = float(atom.GetProp('_GasteigerCharge'))
                if not np.isfinite(val):
                    val = 0.0
            except:
                val = 0.0
            g_charges.append(val)
        desc.append(np.mean(g_charges))
        desc.append(np.std(g_charges))
    except Exception as e:
        desc.append(0.0)
        desc.append(0.0)
    return np.array(desc, dtype=np.float32)

class GraphFeaturizer(MolecularFeaturizer):
    def __init__(self, is_adding_hs=False):
        self.is_adding_hs = is_adding_hs
        super(GraphFeaturizer).__init__()

    def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
        src: List[int] = []
        dest: List[int] = []
        for bond in datapoint.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        return np.asarray([src, dest], dtype=int)
    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        if isinstance(datapoint, Chem.rdchem.Mol):
            if self.is_adding_hs:
                datapoint = Chem.AddHs(datapoint)
        else:
            raise ValueError("Feature field should contain smiles")
    
        mol = datapoint
        smiles = Chem.MolToSmiles(mol)

        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            print(f"{formula} : error ({str(e)})")

        fg_tags = get_functional_group_tags(mol)
        f_atoms: np.ndarray = np.asarray(
            [atom_features(atom, mol, idx, fg_tags) for idx, atom in enumerate(mol.GetAtoms())],
            dtype=float
        )
    
        mol_features = compute_mol_descriptors(mol)

        if len(datapoint.GetBonds()) == 0:
            f_bonds: np.ndarray = np.empty((0, GraphConstants.BOND_FDIM))
        else:
            f_bonds_list = []
            for bond in datapoint.GetBonds():
                b_feat = 2 * [bond_features(bond)]
                f_bonds_list.extend(b_feat)
            f_bonds = np.asarray(f_bonds_list, dtype=float)
        edge_index: np.ndarray = self._construct_bond_index(datapoint)

        arrays_to_check = {
            "node_features": f_atoms,
            "edge_features": f_bonds,
            "edge_index": edge_index,
            "mol_features": mol_features
        }
    
        for name, array in arrays_to_check.items():
            if np.isnan(array).any():
                print(f"\n[ERROR] Molecule {formula} has NaN in {name}!")
                nan_locs = np.argwhere(np.isnan(array))
                print(f"NaN indices in {name}: {nan_locs}")
                print(f"{name} contents:\n{array}")
                raise ValueError(f"NaN detected in {name} for molecule {formula}.")

        return GraphData(node_features=f_atoms,
                         edge_index=edge_index,
                         edge_features=f_bonds,
                         mol_features=mol_features)
