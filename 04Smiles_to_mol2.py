import os
import random
import time
import sys
#
import openbabel
#
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints

from rdkit.ML.Cluster import Butina

print ("\n")
print (" Python:", sys.version )
print (" Numpy :", np.__version__ )
print (" Rdkit :", rdBase.rdkitVersion ,"\n" )

interprete_dict = {'Arg'  : 'R', 'His'  : 'H', 'Lys'  : 'K', 'Asp'  : 'D', 'Glu'  : 'E',
                   'Ser'  : 'S', 'Thr'  : 'T', 'Asn'  : 'N', 'Gln'  : 'Q', 'Cys'  : 'C',
                   'Sec'  : 'U', 'Gly'  : 'G', 'Pro'  : 'P', 'Ala'  : 'A', 'Ile'  : 'I',
                   'Leu'  : 'L', 'Met'  : 'M', 'Phe'  : 'F', 'Trp'  : 'W', 'Tyr'  : 'Y',
                   'Val'  : 'V', 'Dap'  : '1', 'Dab'  : '2', 'BOrn' : '3', 'BLys' : '4',
                   'Hyp'  : 'Z', 'Orn'  : 'O', 'bAla' : '!', 'Gaba' : '?', 'dDap' : '5',
                   'dDab' : '6', 'dBOrn': '7', 'dBLys': '8', 'dArg' : 'r', 'dHis' : 'h',
                   'dLys' : 'k', 'dAsp' : 'd', 'dGlu' : 'e', 'dSer' : 's', 'dThr' : 't',
                   'dAsn' : 'n', 'dGln' : 'q', 'dCys' : 'c', 'dSec' : 'u', 'dGly' : 'g',
                   'dPro' : 'p', 'dAla' : 'a', 'dIle' : 'i', 'dLeu' : 'l', 'dMet' : 'm',
                   'dPhe' : 'f', 'dTrp' : 'w', 'dTyr' : 'y', 'dVal' : 'v', 'dHyp' : 'z',
                   'dOrn' : 'o', 'a5a'  : '=', 'a6a'  : '%', 'a7a'  : '$', 'a8a'  : '@',
                   'Cys1' : 'Ä', 'Cys2' : 'Ö', 'Cys3' : 'Ü', 'dCys1': 'ä', 'dCys2': 'ö',
                   'dCys3': 'ü', 'Ac'   : '&', 'NH2'  : '+', 'met'  : '-', 'cy'   : 'X',
                   'Sar'  : 'é', 'Ase'  : '~', 'Aib'  : 'ÿ', 'Amp'  : 'Ñ', 'dAmp' : 'ñ',
                   'Aza'  : 'Ã', 'dAza' : 'ã', 'Hse'  : '>', 'dHse' : '<', 'Nle'  : 'J',
                   'dNle' : 'j', 'Cit'  : 'B', 'dCit' : 'b', 'a9a'  : '#', 'Acca' : '*' }

interprete_rev_dict = {v: k for k, v in interprete_dict.items()}

# list of possible branching units (1=Dap, 2=Dab, 3=Orn, 4=Lys)
##B = ['1', '2', '3', '4']
# list of possible C-terminals
CT = ['+']
# list of possible N-capping
NT = ['&']
#B4rndm = ['1', '2', '3', '4', '']
#CTrndm = ['+', '']
#NTrndm = ['&', '']
## B4rndm = [''] (only linear generation)
# variables for SMILES generation
B_SMILES = {'1': '[N:2]C(C[N:2])[C:1](O)=O',   '2': '[N:2]C(CC[N:2])[C:1](O)=O',
            '3': '[N:2]C(CCC[N:2])[C:1](O)=O', '4': '[N:2]C(CCCC[N:2])[C:1](O)=O',
            '5': '[N:2]C(C[N:2])[C:1](O)=O',   '6': '[N:2]C(CC[N:2])[C:1](O)=O',
            '7': '[N:2]C(CCC[N:2])[C:1](O)=O', '8': '[N:2]C(CCCC[N:2])[C:1](O)=O' }

AA_SMILES = {'A': '[N:2]C(C)[C:1](O)=O',                'R': '[N:2]C(CCCNC(N)=N)[C:1](O)=O',
             'N': '[N:2]C(CC(N)=O)[C:1](O)=O',          'D': '[N:2]C(CC(O)=O)[C:1](O)=O',
             'C': '[N:2]C(CS)[C:1](O)=O',               'Q': '[N:2]C(CCC(N)=O)[C:1](O)=O',
             'E': '[N:2]C(CCC(O)=O)[C:1](O)=O',         'G': '[N:2]C[C:1](O)=O',
             'H': '[N:2]C(CC1=CNC=N1)[C:1](O)=O',       'I': '[N:2]C(C(C)CC)[C:1](O)=O',
             'K': '[N:2]C(CCCCN)[C:1](O)=O',            'L': '[N:2]C(CC(C)C)[C:1](O)=O',
             'M': '[N:2]C(CCSC)[C:1](O)=O',             'F': '[N:2]C(CC1=CC=CC=C1)[C:1](O)=O',
             'P': 'C1CC[N:2]C1[C:1](O)=O',              'S': '[N:2]C(CO)[C:1](O)=O',
             'T': '[N:2]C(C(O)C)[C:1](O)=O',            'W': '[N:2]C(CC1=CNC2=CC=CC=C12)[C:1](O)=O',
             'Y': '[N:2]C(CC1=CC=C(C=C1)O)[C:1](O)=O',  'V': '[N:2]C(C(C)C)[C:1](O)=O',
             'Ä': '[N:2]C(C[S:1])[C:1](O)=O',           'Ö': '[N:2]C(C[S:2])[C:1](O)=O',
             'Ü': '[N:2]C(C[S:3])[C:1](O)=O',           'Z': 'C1C(O)C[N:2]C1[C:1](O)=O',
             'O': '[N:2]C(CCCN)[C:1](O)=O',             'a': '[N:2]C(C)[C:1](O)=O',
             'r': '[N:2]C(CCCNC(N)=N)[C:1](O)=O',       'ü': '[N:2]C(C[S:3])[C:1](O)=O',
             'n': '[N:2]C(CC(N)=O)[C:1](O)=O',          'd': '[N:2]C(CC(O)=O)[C:1](O)=O',
             'c': '[N:2]C(CS)[C:1](O)=O',               'q': '[N:2]C(CCC(N)=O)[C:1](O)=O',
             'e': '[N:2]C(CCC(O)=O)[C:1](O)=O',         'g': '[N:2]C[C:1](O)=O',
             'h': '[N:2]C(CC1=CNC=N1)[C:1](O)=O',       'i': '[N:2]C(C(C)CC)[C:1](O)=O',
             'k': '[N:2]C(CCCCN)[C:1](O)=O',            'l': '[N:2]C(CC(C)C)[C:1](O)=O',
             'm': '[N:2]C(CCSC)[C:1](O)=O',             'f': '[N:2]C(CC1=CC=CC=C1)[C:1](O)=O',
             'p': 'C1CC[N:2]C1[C:1](O)=O',              's': '[N:2]C(CO)[C:1](O)=O',
             't': '[N:2]C(C(O)C)[C:1](O)=O',            'w': '[N:2]C(CC1=CNC2=CC=CC=C12)[C:1](O)=O',
             'y': '[N:2]C(CC1=CC=C(C=C1)O)[C:1](O)=O',  'v': '[N:2]C(C(C)C)[C:1](O)=O',
             'ä': '[N:2]C(C[S:1])[C:1](O)=O',           'ö': '[N:2]C(C[S:2])[C:1](O)=O',
             '!': '[N:2]CC[C:1](O)=O',                  '?': '[N:2]CCC[C:1](O)=O',
             '=': '[N:2]CCCC[C:1](O)=O',                '%': '[N:2]CCCCC[C:1](O)=O',
             '$': '[N:2]CCCCCC[C:1](O)=O',              '@': '[N:2]CCCCCCC[C:1](O)=O',
             '#': '[N:2]CC[C:1](O)=O',                  '~': '[N:2]C(COC(=O)C)[C:1](O)=O',
             'é': 'C[N:2]C[C:1](O)=O',                  'ÿ': '[N:2]C(C)(C)[C:1](O)=O',
             'B': '[N:2]C(CCCNC(N)=O)[C:1](O)=O',       'b': '[N:2]C(CCCNC(N)=O)[C:1](O)=O',
             'J': '[N:2]C(CCCC)[C:1](O)=O',             'j': '[N:2]C(CCCC)[C:1](O)=O',
             '>': '[N:2]C(CCO)[C:1](O)=O',              '<': '[N:2]C(CCO)[C:1](O)=O',
             'ã': '[N:2]C(CC(=NC=C1O)C=C1)[C:1](O)=O',  'Ã': '[N:2]C(CC(=NC=C1O)C=C1)[C:1](O)=O',
             'ñ': '[N:2]C(CC(=CC=C1N)C=C1)[C:1](O)=O',  'Ñ': '[N:2]C(CC(=CC=C1N)C=C1)[C:1](O)=O',
             '*': '[N:2]CC1CC(C1)[C:1](O)=O' }

T_SMILES = {'+': '[N:2]'}
C_SMILES = {'&': 'C[C:1](=O)'}

#time = 0
# debug
verbose = False

def interprete(seq):
    """translates from 3letters code to one symbol
    Arguments:
        seq {string} -- 3 letters code seq (e.g. Ala-Gly-Leu)
    Returns:
        string -- one letter symbol seq (e.g. AGL)
    """
    new_seq = ''
    seq = seq.split('-')
    for bb in seq:
        new_seq += interprete_dict[bb]
    seq = new_seq
    return seq

def split_seq_components(seq):
    """split seq in generations and branching units
        Arguments:
            seq {string} -- dendrimer sequence

        Returns:
            lists -- generations(gs, from 0 to..), branching units, terminal and capping
    """
    g = []
    gs = []
    bs = []
    t = []
    c = []

    for ix, i in enumerate(seq):
        if i not in ['1', '2', '3', '4', '5', '6', '7', '8']:
            if i in CT:
                t.append(i)
            elif i in NT:
                c.append(i)
            elif i == 'X':
                continue
            elif i == '-':
                if seq[ix - 1] in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    bs.append(i)
                else:
                    g.append(i)
            else:
                g.append(i)
        else:
            gs.append(g[::-1])
            bs.append(i)
            g = []

    gs.append(g[::-1])
    gs = gs[::-1]
    bs = bs[::-1]

    return gs, bs, t, c

def connect_mol(mol1, mol2):
    """it is connecting all Nterminals of mol1 with the Cterminal
        of the maximum possible number of mol2s

        Arguments:
            mol1 {rdKit mol object} -- first molecule to be connected
            mol2 {rdKit mol object} -- second molecule to be connected

        Returns:
            rdKit mol object -- mol1 updated (connected with mol2, one or more)
    """
    # used internally to recognize a methylated aa:
    metbond = False
    # can be set with exclude or allow methylation,
    # it refers to the possibility of having methylation in the entire GA:
    methyl = False

    count = 0

    # detects all the N terminals in mol1
    for atom in mol1.GetAtoms():
        atom.SetProp('Cterm', 'False')
        atom.SetProp('methyl', 'False')
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol2 (it should be one)
    for atom in mol2.GetAtoms():
        atom.SetProp('Nterm', 'False')
        atom.SetProp('methyl', 'False')
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # mol2 is addes to all the N terminal of mol1
    for i in range(count):
        combo = rdmolops.CombineMols(mol1, mol2)
        Nterm = []
        Cterm = []
        # saves in two different lists the index of the atoms which has to be connected
        for atom in combo.GetAtoms():
            if atom.GetProp('Nterm') == 'True':
                Nterm.append(atom.GetIdx())
            if atom.GetProp('Cterm') == 'True':
                Cterm.append(atom.GetIdx())

        # creates the amide bond
        edcombo = rdchem.EditableMol(combo)
        edcombo.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        edcombo.RemoveAtom(Cterm[0] + 1)
        clippedMol = edcombo.GetMol()

        # removes tags and lables form c term atoms which reacted
        clippedMol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        clippedMol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)

        # methylates amide bond
        if metbond == True and methyl == True:
            Nterm = []
            Met = []
            methyl = rdmolfiles.MolFromSmiles('[C:4]')
            for atom in methyl.GetAtoms():
                atom.SetProp('methyl', 'True')
                atom.SetProp('Nterm', 'False')
                atom.SetProp('Cterm', 'False')
            metcombo = rdmolops.CombineMols(clippedMol, methyl)
            for atom in metcombo.GetAtoms():
                if atom.GetProp('Nterm') == 'True':
                    Nterm.append(atom.GetIdx())
                if atom.GetProp('methyl') == 'True':
                    Met.append(atom.GetIdx())
            metedcombo = rdchem.EditableMol(metcombo)
            metedcombo.AddBond(Nterm[0], Met[0], order=Chem.rdchem.BondType.SINGLE)
            clippedMol = metedcombo.GetMol()
            clippedMol.GetAtomWithIdx(Met[0]).SetProp('methyl', 'False')
            clippedMol.GetAtomWithIdx(Met[0]).SetAtomMapNum(0)

        # removes tags and lables form the atoms which reacted
        clippedMol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        clippedMol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)

        # uptades the 'core' molecule
        mol1 = clippedMol
    metbond = False
    return mol1

def attach_capping(mol1, mol2):
    """it is connecting all Nterminals with the desired capping

    Arguments:
        mol1 {rdKit mol object} -- first molecule to be connected
        mol2 {rdKit mol object} -- second molecule to be connected - chosen N-capping

    Returns:
        rdKit mol object -- mol1 updated (connected with mol2, one or more)
    """

    count = 0

    # detects all the N terminals in mol1
    for atom in mol1.GetAtoms():
        atom.SetProp('Cterm', 'False')
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol2 (it should be one)
    for atom in mol2.GetAtoms():
        atom.SetProp('Nterm', 'False')
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # mol2 is addes to all the N terminal of mol1
    for i in range(count):
        combo = rdmolops.CombineMols(mol1, mol2)
        Nterm = []
        Cterm = []

        # saves in two different lists the index of the atoms which has to be connected
        for atom in combo.GetAtoms():
            if atom.GetProp('Nterm') == 'True':
                Nterm.append(atom.GetIdx())
            if atom.GetProp('Cterm') == 'True':
                Cterm.append(atom.GetIdx())

        # creates the amide bond
        edcombo = rdchem.EditableMol(combo)
        edcombo.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        clippedMol = edcombo.GetMol()

        # removes tags and lables form the atoms which reacted
        clippedMol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        clippedMol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        clippedMol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)
        clippedMol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)
        # uptades the 'core' molecule
        mol1 = clippedMol

    return mol1

def cyclize(mol, cy):
    """it is connecting cyclizing the given molecule

    Arguments:
        mol {rdKit mol object} -- molecule to be cyclized
        cy {int} -- 1=yes, 0=no cyclazation

    Returns:
        mols {list of rdKit mol objects} -- possible cyclazation
    """
    count = 0

    # detects all the N terminals in mol
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol (it should be one)
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # detects all the S terminals in mol

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:1]':
            atom.SetProp('Sact1', 'True')
        else:
            atom.SetProp('Sact1', 'False')

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:2]':
            atom.SetProp('Sact2', 'True')
        else:
            atom.SetProp('Sact2', 'False')

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:3]':
            atom.SetProp('Sact3', 'True')
        else:
            atom.SetProp('Sact3', 'False')

    Nterm = []
    Cterm = []
    Sact1 = []
    Sact2 = []
    Sact3 = []

    # saves active Cysteins postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact1') == 'True':
            Sact1.append(atom.GetIdx())

    # saves active Cysteins 2 postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact2') == 'True':
            Sact2.append(atom.GetIdx())

    # saves active Cysteins 3 postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact3') == 'True':
            Sact3.append(atom.GetIdx())

    # creates the S-S bond (in the current version only two 'active' Cys, this codo picks two random anyway):
    while len(Sact1) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact1)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact1[x]
        b = Sact1[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact1', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact1', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact1.remove(a)
        Sact1.remove(b)

    while len(Sact2) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact2)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact2[x]
        b = Sact2[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact2', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact2', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact2.remove(a)
        Sact2.remove(b)

    while len(Sact3) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact3)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact3[x]
        b = Sact3[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact3', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact3', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact3.remove(a)
        Sact3.remove(b)

    # saves active C and N terminals postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Nterm') == 'True':
            Nterm.append(atom.GetIdx())
        if atom.GetProp('Cterm') == 'True':
            Cterm.append(atom.GetIdx())

    if cy == 1:
        edmol = rdchem.EditableMol(mol)

        # creates the amide bond
        edmol.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        edmol.RemoveAtom(Cterm[0] + 1)

        mol = edmol.GetMol()

        # removes tags and lables form the atoms which reacted
        mol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        mol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        mol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)
        mol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)

    return mol

def smiles_from_seq(seq):
    """Calculates the smiles of a given peptide dendrimer sequence

        Arguments:
            seq {string} -- peptide dendrimer sequence
        Returns:
            string -- molecule_smile - SMILES of the peptide
    """
    gs, bs, terminal, capping = split_seq_components(seq)
    # modifies the Cterminal
    if terminal:
        molecule = rdmolfiles.MolFromSmiles(T_SMILES[terminal[0]])
    else:
        molecule = ''

    # creates the dendrimer structure
    for gen in gs:
        for aa in gen:
            if aa == '-':
                metbond = True
                continue
            if molecule == '':
                molecule = rdmolfiles.MolFromSmiles(AA_SMILES[aa])
            else:
                molecule = connect_mol(molecule, rdmolfiles.MolFromSmiles(AA_SMILES[aa]))

        if bs:
            if bs[0] == '-':
                metbond = True
                bs.pop(0)
            if molecule == '':
                molecule = rdmolfiles.MolFromSmiles(B_SMILES[bs[0]])
            else:
                molecule = connect_mol(molecule, rdmolfiles.MolFromSmiles(B_SMILES[bs[0]]))
            bs.pop(0)

    # adds capping to the N-terminal (the called clip function is different, cause the listed smiles
    # for the capping are already without OH, it is not necessary removing any atom after foming the new bond)
    if capping and molecule != '':
        molecule = attach_capping(molecule, rdmolfiles.MolFromSmiles(C_SMILES[capping[0]]))
    # clean the smile from all the tags
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    molecule_smile = rdmolfiles.MolToSmiles(molecule, isomericSmiles=True).replace('[N]', 'N').replace('[C]', 'C')
    return molecule_smile

def smiles_from_seq_cyclic(seq):
    """Calculates the smiles of the given peptide sequence and cyclize it
        Arguments:
            seq {string} -- peptide dendrimer sequence
        Returns:
            string -- molecule_smile - SMILES of the peptide
    """
    # used internally to recognize a methylated aa:
    metbond = False
    # can be set with exclude or allow methylation,
    # it refers to the possibility of having methylation in the entire GA:
    methyl = False

    if 'X' in seq:
        cy = 1
        for i in NT:
            seq = seq.replace(i, '')
        for i in CT:
            seq = seq.replace(i, '')
    else:
        cy = 0

    gs, bs, terminal, capping = split_seq_components(seq)

    # modifies the Cterminal
    if terminal:
        molecule = rdmolfiles.MolFromSmiles(T_SMILES[terminal[0]])
    else:
        molecule = ''

    if bs:
        if verbose:
            print('dendrimer, cyclization not possible, branching unit will not be considered')

    # creates the linear peptide structure
    for gen in gs:
        for aa in gen:
            if aa == 'X':
                continue
            if aa == '-':
                metbond = True
                continue
            if molecule == '':
                molecule = rdmolfiles.MolFromSmiles(AA_SMILES[aa])
            else:
                molecule = connect_mol(molecule, rdmolfiles.MolFromSmiles(AA_SMILES[aa]))

    # adds capping to the N-terminal (the called clip function is different, cause the listed smiles
    # for the capping are already without OH, it is not necessary removing any atom after foming the new bond)
    if capping:
        molecule = attach_capping(molecule, rdmolfiles.MolFromSmiles(C_SMILES[capping[0]]))

    # cyclize
    if molecule == '':
        smiles = ''
        return smiles

    #print (cy)
    molecule = cyclize(molecule, cy)

    # clean the smile from all the tags
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = rdmolfiles.MolToSmiles(molecule, isomericSmiles=True).replace('[N]', 'N').replace('[C]', 'C')

    return smiles

def gen_conformers(mol, numConfs):
    """Improving Conformer Generation for Small Rings and Macrocycles
    Based on Distance Geometry and Experimental Torsional-Angle
    Preferences. DOI: 10.1021/acs.jcim.0c00025"""
    # ETDG, ETKDG, ETKDGv2, ETKDGv3
    cids = AllChem.EmbedMultipleConfs(mol,
                                    numConfs=numConfs,
                                    params=AllChem.ETKDGv3())
    ##cids = AllChem.EmbedMultipleConfs(mol,
    ##                                clearConfs=True,
    ##                                numConfs=100,
    ##                                pruneRmsThresh=1)

    ##ids = AllChem.EmbedMultipleConfs(mol,numConfs=numConfs,axAttempts=maxAttempts,
    ##                                pruneRmsThresh=pruneRmsThresh,
    ##                                useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
    ##                                useBasicKnowledge=useBasicKnowledge,
    ##                                enforceChirality=enforceChirality,
    ##                                numThreads=0)"""

    ##print (len(cids));
    ##print(mol.GetNumConformers())
    return list(cids)

def calc_energy(mol, conformerId, minimizeIts):
	ff = AllChem.MMFFGetMoleculeForceField(mol,
                                        AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94s'),
                                        confId=conformerId)
	ff.Initialize()
	ff.CalcEnergy()
	results = {}
	if minimizeIts > 0:
		results["converged"] = ff.Minimize(maxIts=minimizeIts)
	results["energy_abs"] = ff.CalcEnergy()
	return results

def cluster_conformers(mol, mode="RMSD", threshold=2.0):
	if mode == "TFD":
		dmat = TorsionFingerprints.GetTFDMatrix(mol)
	else:
		dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
	rms_clusters = Butina.ClusterData(dmat,
                                    mol.GetNumConformers(),
                                    threshold,
                                    isDistData=True,
                                    reordering=True)
	return rms_clusters

def align_conformers(mol, clust_ids):
	rmslist = []
	AllChem.AlignMolConformers(mol, confIds=clust_ids, RMSlist=rmslist)
	return rmslist

def write_conformers_to_sdf(mol, filename, rmsClusters, conformerPropsDict, minEnergy):
	w = Chem.SDWriter(filename)
	for cluster in rmsClusters:
		for confId in cluster:
			for name in mol.GetPropNames():
				mol.ClearProp(name)
			conformerProps = conformerPropsDict[confId]
			mol.SetIntProp("conformer_id", confId + 1)
			for key in conformerProps.keys():
				mol.SetProp(key, str(conformerProps[key]))
			e = conformerProps["energy_abs"]
			if e:
				mol.SetDoubleProp("energy_delta", e - minEnergy)
			w.write(mol, confId=confId)
	w.flush()
	w.close()

def peptide_smile_FF(peptide, numMol):
    # parameters
    numConfs = 5
    # cluster method: (RMSD|TFD) = RMSD
    clusterMethod = "RMSD"
    clusterThreshold = 2.0
    minimizeIterations = 20

    seq = interprete(peptide)
    #
    #smi = smiles_from_seq(seq)
    smi = smiles_from_seq_cyclic(seq)
    #print (smi)
    #
    mol = Chem.MolFromSmiles(smi)
    m   = Chem.AddHs(mol)
    # generate the confomers
    conformerIds = gen_conformers(m, numConfs)
    conformerPropsDict = {}
    for conformerId in conformerIds:
        # energy minimise (optional) and energy calculation
        props = calc_energy(m, conformerId, minimizeIterations)
        conformerPropsDict[conformerId] = props

    # cluster the conformers
    rmsClusters = cluster_conformers(m, clusterMethod, clusterThreshold)
    print ("Molecule", peptide, ": generated", len(conformerIds), "conformers and", len(rmsClusters), "clusters")
    #
    rmsClustersPerCluster = []
    clusterNumber         = 0
    minEnergy             = 9999999999999
    minID                 = 0
    minCluster            = 0
    for cluster in rmsClusters:
        ##print (cluster)
        clusterNumber = clusterNumber+1
        rmsWithinCluster = align_conformers(m, cluster)
        for conformerId in cluster:
            e = props["energy_abs"]
            if e < minEnergy:
                minEnergy = e
                minID     = cluster[0]
            ##print (str(e) + " " + str (conformerId) + " " + str (clusterNumber))
            props = conformerPropsDict[conformerId]
            props["cluster_no"] = clusterNumber
            props["cluster_centroid"] = cluster[0] + 1
            idx = cluster.index(conformerId)
            if idx > 0:
                props["rms_to_centroid"] = rmsWithinCluster[idx-1]
            else:
                props["rms_to_centroid"] = 0.0

    print ( "Minimum energy: " + str(minEnergy) + " ID: " + str(minID))
    ## Save olny one file.sdf
    filename = str(peptide) + ".sdf"
    w = Chem.SDWriter(filename)

    for cluster in rmsClusters:
        for confId in cluster:
            conformerPropsTmp = conformerPropsDict[confId]
            enel = conformerPropsTmp["energy_abs"]
            if (enel == minEnergy):
                ##print ( str(enel) + " " + str(minEnergy))
                for name in m.GetPropNames():
                    m.ClearProp(name)
                conformerProps = conformerPropsDict[confId]
                m.SetIntProp("conformer_id", confId + 1)
                for key in conformerProps.keys():
                    m.SetProp(key, str(conformerProps[key]))
                e = conformerProps["energy_abs"]
                if e:
                    m.SetDoubleProp("energy_delta", e - minEnergy)
                w.write(m, confId=confId)
    w.flush()
    w.close()



    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf","mol2")
    mol= openbabel.OBMol()
    obConversion.ReadFile(mol,filename)
    obConversion.WriteFile(mol,filename[:-4]+".mol2")

    # All conformers
    ##write_conformers_to_sdf(m, str(peptide) +"_" +str(numMol) + ".sdf", rmsClusters, conformerPropsDict, minEnergy)








if __name__=='__main__':

    t1=time.time()
    # Read file
    file_name = "03Filter_Cyc_200.csv"

    FI = open(file_name, 'r')
    count = 0
    for line in FI:
        line = line.replace('\n', '').replace('\r', '')
        parameter=line.split(sep="\t",maxsplit=3)
        seq = str(parameter[0])
        peptide_smile_FF(seq,count)
        count = count + 1

    FI.close()
    #
    print ("Fin programa")
    #
    t2=time.time()
    resTime1=(t2-t1)
    print (' Reading took %.2f seconds. \n' % resTime1 )
