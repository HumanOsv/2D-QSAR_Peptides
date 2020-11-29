import os
import random
import subprocess as sub
import time
import sys

from multiprocessing import Process

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops

## Pendientes
## glycosylated serine
## Acpc (1-aminocyclopentane carboxylic acid)
## Me-Gly (N,N dimethylglycine)
## Bet (betaine, 1-carboxy-N,N,N-trimethyl methanaminium hydroxide)
## Suc (succinic acid)

## Añadido al Original
##
## Acca (*)
## Amp (p-Amino-L-phenylalanine) ('ñ')
## Aza (Azatyrosine) ('ã')
## Hse (homoserine) ('<')
## Aib (2-aminoisobutyric acid) ('ÿ')
## Nle (norleucine) ('j')
## Cit (citrulline) ('b')
## Sar (sarcosine) ('é')
## Ase (Acetylserine) ('~')
##
## Oroginal
## Orn  Ornithine
## Hyp  Hydroxyproline
## bAla Beta-alanine
## Gaba Gamma-aminobutyric acid
## a5a  Delta-aminopentanoic acid
## a6a  Epsilon-aminohexanoic acid
## a7a  Zeta-aminoheptanoic acid
## a8a  Eta-aminooctanoic acid
## a9a  Theta-aminononaanoic acid
## Dap  2,3-diaminopropionic acid as branching unit
## Dab  2,4-diaminobutyric acid as branching unit
## BOrn Ornithine as branching unit
## BLys Lysine as branching unit
## cy   Head-to-tail cyclization. It is always placed at the beginning (left, N terminus) of the sequence.
## Cys1 First pair of cyclizes cysteines. Always in pair, never next to each other.
## Cys2 Second pair of cyclizes cysteines. They are always present in pair, never next to each other, present only if Cys1 is already part of the sequence.
## Cys3 Third pair of cyclizes cysteines. They are always present in pair, never next to each other, present only if Cys1 and Cys2 are already part of the sequence.
## Ac   N-terminus acetylation. It is always placed at the beginning (N-terminus, left) of the sequence
## NH2  C-terminus amide. It is always placed at the end (C-terminus, right) of the sequence

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

def write_file(array,num):

    fw = open( 'ValuesCyc' + str(num) + '.txt', 'w')
    for peptide in array:
        seq = interprete(peptide)
    ##    smi = smiles_from_seq(seq)
        smi = smiles_from_seq_cyclic(seq)
        result = (str(peptide) + "\t" + str(smi) + "\n" )
        ##print (result)
        fw.write(result)

    fw.close()



if __name__=='__main__':

    t1=time.time()

    # Read file
    ##splitLen = 109016        # 20 lines per file
    splitLen = 4273293
    outputBase = 'output' # output.1.txt, output.2.txt, etc.
    # This is shorthand and not friendly with memory
    # on very large files (Sean Cavanagh), but it works.
    input = open('Data_peptidos_Lin-42732900.txt', 'r').read().split('\n')
    ##input = open('Data_peptidos_Lin-42732900.txt', 'r').read().split('\n')
    at = 1
    for lines in range(0, len(input), splitLen):
        # First, get the list slice
        outputData = input[lines:lines+splitLen]
        # Now open the output file, join the new slice with newlines
        # and write it out. Then close the file.
        output = open(outputBase + str(at) + '.txt', 'w')
        output.write('\n'.join(outputData))
        output.close()
        # Increment the counter
        at += 1
    print ("\nFinal dividir archivo principal\n")
    # Read Smiles
    seqAA_1 = []
    seqAA_2 = []
    seqAA_3 = []
    seqAA_4 = []
    seqAA_5 = []
    seqAA_6 = []
    seqAA_7 = []
    seqAA_8 = []
    seqAA_9 = []
    seqAA_10 = []
    #
    filename_1 = "output1.txt"
    f_1=open(filename_1, "r")
    f1_1 = f_1.readlines()
    f_1.close()
    for x in f1_1:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_1.append(x)

    filename_2 = "output2.txt"
    f_2=open(filename_2, "r")
    f1_2 = f_2.readlines()
    f_2.close()
    for x in f1_2:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_2.append(x)

    filename_3 = "output3.txt"
    f_3=open(filename_3, "r")
    f1_3 = f_3.readlines()
    f_3.close()
    for x in f1_3:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_3.append(x)

    filename_4 = "output4.txt"
    f_4=open(filename_4, "r")
    f1_4 = f_4.readlines()
    f_4.close()
    for x in f1_4:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_4.append(x)

    filename_5 = "output5.txt"
    f_5=open(filename_5, "r")
    f1_5 = f_5.readlines()
    f_5.close()
    for x in f1_5:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_5.append(x)

    filename_6 = "output6.txt"
    f_6=open(filename_6, "r")
    f1_6 = f_6.readlines()
    f_6.close()
    for x in f1_6:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_6.append(x)

    filename_7 = "output7.txt"
    f_7=open(filename_7, "r")
    f1_7 = f_7.readlines()
    f_7.close()
    for x in f1_7:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_7.append(x)

    filename_8 = "output8.txt"
    f_8=open(filename_8, "r")
    f1_8 = f_8.readlines()
    f_8.close()
    for x in f1_8:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_8.append(x)

    filename_9 = "output9.txt"
    f_9=open(filename_9, "r")
    f1_9 = f_9.readlines()
    f_9.close()
    for x in f1_9:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_9.append(x)

    filename_10 = "output10.txt"
    f_10=open(filename_10, "r")
    f1_10 = f_10.readlines()
    f_10.close()
    for x in f1_10:
        # delete \n
        x = x.replace('\n', '').replace('\r', '')
        seqAA_10.append(x)

    print ("Fin leer output.txt");

    p1 = Process(target=write_file, args=(seqAA_1,1))
    p2 = Process(target=write_file, args=(seqAA_2,2))
    p3 = Process(target=write_file, args=(seqAA_3,3))
    p4 = Process(target=write_file, args=(seqAA_4,4))
    p5 = Process(target=write_file, args=(seqAA_5,5))
    p6 = Process(target=write_file, args=(seqAA_6,6))
    p7 = Process(target=write_file, args=(seqAA_7,7))
    p8 = Process(target=write_file, args=(seqAA_8,8))
    p9 = Process(target=write_file, args=(seqAA_9,9))
    p10= Process(target=write_file, args=(seqAA_10,10))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()

    print ("Fin paralelo")
    #
    print ("Fin programa")
    #
    t2=time.time()
    resTime1=(t2-t1)
    print (' Reading took %.2f seconds. \n' % resTime1 )
