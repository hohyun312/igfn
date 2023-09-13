from rdkit.Chem.rdchem import BondType
from rdkit import Chem
import networkx as nx


class MolGraph:
    id2atom = ["C", "N", "O", "S", "P", "F", "I", "Cl", "Br"]
    id2bond = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]  # BondType.AROMATIC

    bond2id = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    atom2id = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "F": 5, "I": 6, "Cl": 7, "Br": 8}

    num_edge_types = len(bond2id)
    num_node_types = len(atom2id)

    @classmethod
    def from_molecule(cls, molecule):
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        node_type = cls.get_node_type(molecule)
        edge_list = cls.get_edge_list(molecule)

        G = nx.Graph()
        for i, nt in enumerate(node_type):
            G.add_node(i, node_type=nt)

        for i, j, et in edge_list:
            G.add_edge(i, j, edge_type=et)

        return G

    @classmethod
    def to_molecule(cls, molg):
        mol = Chem.RWMol()
        for i, attr in molg.nodes(data=True):
            atom_symbol = cls.id2atom[attr["node_type"]]
            mol.AddAtom(Chem.Atom(atom_symbol))

        for i, j, attr in molg.edges(data=True):
            edge_type = cls.id2bond[attr["edge_type"]]
            mol.AddBond(i, j, order=edge_type)
        return mol

    @classmethod
    def from_smiles(cls, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        return cls.from_molecule(molecule)

    @classmethod
    def get_node_type(cls, molecule):
        node_type = []
        for i in range(molecule.GetNumAtoms()):
            atom = molecule.GetAtomWithIdx(i)
            node_type.append(cls.atom2id[atom.GetSymbol()])
        return node_type

    @classmethod
    def get_edge_list(cls, molecule):
        edge_list = []
        for i in range(molecule.GetNumBonds()):
            bond = molecule.GetBondWithIdx(i)
            t = cls.bond2id[bond.GetBondType()]
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [[u, v, t], [v, u, t]]
        return edge_list

    def __repr__(self):
        return "%s(num_node=%s, num_edge=%s)" % (
            self.__class__.__name__,
            self.num_node,
            self.num_edge,
        )
