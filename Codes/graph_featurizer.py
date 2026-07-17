import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Sequence, Set
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.utils.molecule_feature_utils import one_hot_encode

def modify_molecule_for_test(smiles: str, action: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    if action == "saturate":
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        Chem.SanitizeMol(mol)
    elif action == "reduce_aldehyde":
        rxn = AllChem.ReactionFromSmarts('[CX3H1:1](=O)[#6:2]>>[CX4H2:1](O)[#6:2]')
        ps = rxn.RunReactants((mol,))
        if ps: mol = ps[0][0]
    return Chem.MolToSmiles(mol)

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

class GraphConstants:
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES = {
        'valence': [0, 1, 2, 3, 4, 5, 6],
        'degree': [0, 1, 2, 3, 4, 5],
        'num_Hs': [0, 1, 2, 3, 4],
        'formal_charge': [-1, -2, 1, 2, 3, 0],
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'hybridization': ["SP", "SP2", "SP3"]
    }
    BOND_FDIM = 7

def atom_features(atom: Chem.Atom, 
                  mol: Chem.Mol, 
                  atom_idx: int, 
                  fg_tags: Optional[Dict[str, Set[int]]] = None) -> List[float]:
    if atom is None:
        expected_dim = 132 + (len(SMARTS_GROUPS) if fg_tags is not None else 0)
        return [0.0] * expected_dim
    
    features = []
    features += one_hot_encode(atom.GetTotalValence(), GraphConstants.ATOM_FEATURES['valence'])
    features += one_hot_encode(atom.GetTotalDegree(), GraphConstants.ATOM_FEATURES['degree'])
    features += one_hot_encode(atom.GetTotalNumHs(), GraphConstants.ATOM_FEATURES['num_Hs'])
    features += one_hot_encode(atom.GetFormalCharge(), GraphConstants.ATOM_FEATURES['formal_charge'])
    features += one_hot_encode(atom.GetAtomicNum()-1, GraphConstants.ATOM_FEATURES['atomic_num'])
    features += [float(atom.GetIsAromatic()), float(atom.IsInRing()), float(atom.HasProp("_ChiralityPossible"))]
    features += one_hot_encode(str(atom.GetHybridization()), GraphConstants.ATOM_FEATURES['hybridization'])
    
    try:
        gasteiger = float(atom.GetProp('_GasteigerCharge'))
        features.append(gasteiger if np.isfinite(gasteiger) else 0.0)
    except:
        features.append(0.0)
    features.append(atom.GetMass() * 0.01) 
                      
    if fg_tags is not None:
        for name in SMARTS_GROUPS:
            features.append(1.0 if atom_idx in fg_tags[name] else 0.0)
            
    return features

class GraphFeaturizer(MolecularFeaturizer):
    def __init__(self, is_adding_hs: bool = False, use_fg_features: bool = True):
        self.is_adding_hs = is_adding_hs
        self.use_fg_features = use_fg_features
        super().__init__()

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        if not isinstance(datapoint, Chem.Mol):
            raise ValueError("Input must be an RDKit Mol object.")

        mol = Chem.AddHs(datapoint) if self.is_adding_hs else datapoint
        
        try:
            mol.UpdatePropertyCache()
            AllChem.ComputeGasteigerCharges(mol)
        except:
            pass

        fg_tags = None
        if self.use_fg_features:
            fg_tags = {name: set() for name in SMARTS_GROUPS}
            for name, pattern in SMARTS_PATTERNS.items():
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        fg_tags[name].update(match)

        f_atoms = np.asarray([
            atom_features(atom, mol, i, fg_tags) 
            for i, atom in enumerate(mol.GetAtoms())
        ], dtype=np.float32)

        if mol.GetNumBonds() == 0:
            edge_index = np.empty((2, 0), dtype=int)
            edge_features = np.empty((0, GraphConstants.BOND_FDIM), dtype=np.float32)
        else:
            edge_indices, edge_feats = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
                bt = bond.GetBondType()
                b_feat = [
                    0.0, 
                    float(bt == Chem.rdchem.BondType.SINGLE),
                    float(bt == Chem.rdchem.BondType.DOUBLE),
                    float(bt == Chem.rdchem.BondType.TRIPLE),
                    float(bt == Chem.rdchem.BondType.AROMATIC),
                    float(bond.GetIsConjugated()),
                    float(bond.IsInRing())
                ]
                edge_feats.extend([b_feat, b_feat])
            
            edge_index = np.array(edge_indices, dtype=int).T
            edge_features = np.array(edge_feats, dtype=np.float32)
            
        return GraphData(
            node_features=f_atoms,
            edge_index=edge_index,
            edge_features=edge_features
        )
