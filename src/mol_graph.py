import pickle
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from src.containers import State, StateType

class MolGraph(State):
    
    @classmethod
    def build_vocab(cls, smiles, min_count=100):
        invalid_smiles = 0
        mols = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                mols.append(mol)
            except:
                invalid_smiles += 1
        

        max_num_nodes = 0
        max_num_edges = 0
        max_degree = 0
        count = defaultdict(int)
        atom2valence = {"UNK": 4}
        for mol in mols:
            for atom in mol.GetAtoms():
                key = (atom.GetSymbol(), atom.GetFormalCharge())
                count[key] += 1
                atom2valence[key] = atom.GetExplicitValence() + atom.GetImplicitValence()
                max_degree = max(max_degree, atom.GetDegree())
            max_num_nodes = max(max_num_nodes, mol.GetNumAtoms())
            max_num_edges = max(max_num_edges, 2 * mol.GetNumBonds())
        
        id2atom = ["UNK"]
        for k, v in count.items():
            if v > min_count:
                id2atom.append(k)

        cls.id2atom = id2atom
        cls.id2bond = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
        cls.atom2id = dict([(v, i) for i, v in enumerate(cls.id2atom)])
        cls.bond2id = dict([(v, i) for i, v in enumerate(cls.id2bond)])
        cls.num_edge_types = len(cls.bond2id)
        cls.num_node_types = len(cls.atom2id)
        cls.id2valence = [atom2valence[cls.id2atom[i]] for i in range(cls.num_node_types)]
        cls.max_degree = max_degree
        cls.max_num_nodes = max_num_nodes
        cls.max_num_edges = max_num_edges

    @classmethod
    def save_vocab(cls, path):
        data = {
            "id2valence": cls.id2valence, 
            "id2atom": cls.id2atom,
            "max_degree": cls.max_degree,
            "max_num_nodes": cls.max_num_nodes,
            "max_num_edges": cls.max_num_edges
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_vocab(cls, path):
        with open(path, "rb") as f:
             data = pickle.load(f)
        cls.id2valence = data["id2valence"]
        cls.id2atom = data["id2atom"]
        cls.id2bond = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
        cls.atom2id = dict([(v, i) for i, v in enumerate(cls.id2atom)])
        cls.bond2id = dict([(v, i) for i, v in enumerate(cls.id2bond)])
        cls.num_edge_types = len(cls.bond2id)
        cls.num_node_types = len(cls.atom2id)
        cls.max_degree = data["max_degree"]
        cls.max_num_nodes = data["max_num_nodes"]
        cls.max_num_edges = data["max_num_edges"]

    @classmethod
    def get_node_type(cls, molecule):
        node_type = []
        for i in range(molecule.GetNumAtoms()):
            atom = molecule.GetAtomWithIdx(i)
            key = (atom.GetSymbol(), atom.GetFormalCharge())
            node_type.append(cls.atom2id.get(key, 0))
        return node_type
    
    @classmethod
    def get_edge_list_and_type(cls, molecule):
        edge_list = []
        edge_type = []
        for i in range(molecule.GetNumBonds()):
            bond = molecule.GetBondWithIdx(i)
            t = cls.bond2id[bond.GetBondType()]
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [(u, v), (v, u)]
            edge_type += [t, t]
        return edge_list, edge_type
    
    @classmethod
    def from_mol(cls, molecule):
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        node_type = cls.get_node_type(molecule)
        edge_list, edge_type = cls.get_edge_list_and_type(molecule)
        return cls(StateType.Terminal, node_type, edge_type, edge_list)
    
    @classmethod
    def from_smiles(cls, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        return cls.from_mol(molecule)
    
    def to_mol(self):
        mol = Chem.RWMol()
        for t in self.node_type:
            if t != 0:
                symbol, charge = self.id2atom[t]
                atom = Chem.Atom(symbol)
                atom.SetFormalCharge(charge)
                mol.AddAtom(atom)
            else:
                raise ValueError("Unknown node type")
    
        # remove duplicate edges
        edge_list, edge_type = self.edge_list[::2], self.edge_type[::2]
        for (i, j), t in zip(edge_list, edge_type):
            mol.AddBond(i, j, order=self.id2bond[t])
        return mol

    def expl_valence(self):
        return [
            sum([1 + self.e2t[(u, v)] for v in u_neighbors]) 
                for u, u_neighbors in enumerate(self.adj)
        ]

    def total_valence(self):
        return [self.valence[nid] for nid in self.node_type]
        


if __name__ == "__main__":
    import pandas as pd
    print("Buliding vocab...")

    df = pd.read_csv("./data/250k_rndm_zinc_drugs_clean_3.csv")
    smiles = df.smiles.tolist()
    MolGraph.build_vocab(smiles)
    MolGraph.save_vocab("./data/vocab.pkl")

    print("Saved at ./data/vocab.pkl")
else:
    MolGraph.load_vocab("./data/vocab.pkl")