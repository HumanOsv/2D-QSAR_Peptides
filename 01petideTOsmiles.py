import os
import time
import sys
import math
import gzip
import pickle
import numpy as np
#
from collections import defaultdict
# Biopython
from Bio import SeqIO
# rdkit cheminformania
from rdkit import DataStructs
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors
#
from rdkit.Chem.Draw import SimilarityMaps
#Machine learning modules
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



#from sklearn import preprocessing
#import scipy
#from sklearn.metrics import mean_squared_error
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import r2_score

#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import PandasTools

#from rdkit.Chem import rdmolfiles
#from rdkit.Chem import rdmolops
#from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import AllChem
#from rdkit.Chem import Descriptors
















print ("\n")
print (" Python:", sys.version )
print (" Numpy :", np.__version__ )
print (" Rdkit :", rdBase.rdkitVersion ,"\n" )

_fscores = None



def genFP(mol,Dummy=-1):
    # Helper function to convert to Morgan type fingerprint in Numpy Array
    fp = SimilarityMaps.GetMorganFingerprint(mol)
    fp_vect = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fp_vect)
    return fp_vect

def readFragmentScores(name='fpscores'):
    #import cPickle,gzip
    global _fscores
    _fscores = pickle.load(gzip.open('%s.pkl.gz'%name))
    outDict = {}
    for i in _fscores:
        for j in range(1,len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol,ri=None):
    if ri is None:
        ri=mol.GetRingInfo()
    arings = [set(x) for x in ri.AtomRings()]
    spiros=set()
    for i,ari in enumerate(arings):
        for j in range(i+1,len(arings)):
            shared=ari&arings[j]
            if len(shared)==1:
                spiros.update(shared)
    nSpiro=len(spiros)
    # find bonds that are shared between rings that share at least 2 bonds:
    nBridge=0
    brings = [set(x) for x in ri.BondRings()]
    bridges=set()
    for i,bri in enumerate(brings):
        for j in range(i+1,len(brings)):
            shared=bri&brings[j]
            if len(shared)>1:
                atomCounts=defaultdict(int)
                for bi in shared:
                    bond = mol.GetBondWithIdx(bi)
                    atomCounts[bond.GetBeginAtomIdx()]+=1
                    atomCounts[bond.GetEndAtomIdx()]+=1
                tmp=0
                for ai,cnt in atomCounts.items():
                    if cnt==1:
                        tmp+=1
                        bridges.add(ai)
                    #if tmp!=2: # no need to stress the users
                        #print 'huh:',tmp
    return len(bridges),nSpiro

def calculateScore(m):
    if _fscores is None: readFragmentScores()
    """Ertl, P. and Schuffenhauer A. “Estimation of Synthetic Accessibility
    Score of Drug-like Molecules based on Molecular Complexity and Fragment
    Contributions” Journal of Cheminformatics 1:8 (2009)"""
    #
    # fragment score
    #<- 2 is the *radius* of the circular fingerprint
    fp = rdMolDescriptors.GetMorganFingerprint(m,2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId,v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp,-4)*v
    score1 /= nf
    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m,includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads,nSpiro=numBridgeheadsAndSpiro(m,ri)
    nMacrocycles=0
    for x in ri.AtomRings():
        if len(x)>8: nMacrocycles+=1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters+1)
    spiroPenalty = math.log10(nSpiro+1)
    bridgePenalty = math.log10(nBridgeheads+1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0: macrocyclePenalty = math.log10(2)

    score2 = 0. -sizePenalty -stereoPenalty -spiroPenalty -bridgePenalty -macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3
    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.: sascore = 8. + math.log(sascore+1.-9.)
    if sascore > 10.: sascore = 10.0
    elif sascore < 1.: sascore = 1.0

    return sascore

def pepLinealtoSMILE(seq):
    # Convertir Fasta a SMILES
    tmpSeq = seq[0:1]+"."+seq[1:2]+"."+seq[2:3]+"."+seq[3:4]+"."+seq[4:5]+"."+seq[5:6]+"."+seq[6:7]
    helmLineal="PEPTIDE1{"+tmpSeq +"}$$$$V2.0"
    SeqFasta = Chem.MolFromHELM(str(helmLineal))
    SeqSmiles=Chem.MolToSmiles(SeqFasta)
    #
    #print (SeqSmiles)
    return SeqSmiles

def pepCyclictoSMILE(seq):
    # Convertir Fasta a SMILES
    tmpSeq = seq[0:1]+"."+seq[1:2]+"."+seq[2:3]+"."+seq[3:4]+"."+seq[4:5]+"."+seq[5:6]+"."+seq[6:7]
    helmCyclic ="PEPTIDE1"+"{"+tmpSeq +"}$PEPTIDE1,PEPTIDE1,7:R2-1:R1$$$"
    SeqFasta = Chem.MolFromHELM(str(helmCyclic))
    SeqSmiles=Chem.MolToSmiles(SeqFasta)
    #
    return SeqSmiles

def QSArproperties(mol,sa, seq, Ames):
    propQSAR=QED.properties(mol)
    """Bickerton, G.R.; Paolini, G.V.; Besnard, J.; Muresan, S.; Hopkins, A.L. (2012)
    Quantifying the chemical beauty of drugs,
    Nature Chemistry, 4, 90-98
    [https://doi.org/10.1038/nchem.1243]"""
    #
    """Wildman, S.A.; Crippen, G.M. (1999)
    Prediction of Physicochemical Parameters by Atomic Contributions,
    Journal of Chemical Information and Computer Sciences, 39, 868-873
    [https://doi.org/10.1021/ci990307l]"""
    #
    MolWeight=propQSAR.MW
    MolLogP=propQSAR.ALOGP
    HbondA=propQSAR.HBA
    HbondD=propQSAR.HBD
    PolarSA=propQSAR.PSA
    Rbonds=propQSAR.ROTB
    Aromatic=propQSAR.AROM
    MolarRefractivity=Crippen.MolMR(mol)
    SynthAcces=sa
    nAtoms = mol.GetNumAtoms()
    AmesMutagenic = Ames
    #print ("" + str(MolWeight) + " " + str(MolLogP) + " " +str(SynthAcces) )


if __name__=='__main__':
    #
    # Time
    t1=time.time()
    # Input DataBase
    fileFasta="Permutacion-Ang-1-7.fasta"
    # Data Base Synthetic Accessibility
    readFragmentScores("fpscores")
    """Hansen, K., Mika, S., Schroeter, T., Sutter, A., ter Laak, A.,
    Steger-Hartmann, T., Heinrich, N., and Muller, K. R. (2009)
    Benchmark Data Set for in Silico Prediction of Ames Mutagenicity.
    J. Chem. Inf. Model. 49, 2077−2081."""
    data = np.genfromtxt('smiles_cas_N6512.smi',
                        delimiter='\t',
                        names=['Smiles','CAS','Mutagen'],
                        encoding=None,
                        dtype=None,
                        comments='##')
    #
    # Convert smiles to RDkit molecules and calculate fingerprints
    mols = []
    X    = []
    y    = []
    for record in data:
        try:
            mol = Chem.MolFromSmiles(record[0])
            if type(mol) != type(None):
                fp_vect = genFP(mol)
                mols.append([mol, record[1],record[2]])
                X.append(fp_vect)
                y.append(record[2])
        except:
            print ("Failed for CAS: %s" % record[1])
    #See how succesful the conversions were
    print ("Imported smiles %s" % len(data))
    print ("Converted smiles %s" % len(mols))
    # Prepare the data for modelling
    X=np.array(X)
    y=np.array(y)
    #
    # Random Forest
    print ('\n <- Random Forest -> \n')
    # Cross Validate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # Prepare scoring lists
    accs_train = []
    accs_test  = []
    for i in [1, 3, 5, 10, 30, 50, 100, 500, 1000]:
        forest = RandomForestClassifier(n_estimators=1000, max_depth=i,n_jobs=-1)
        # Fit and score
        forest.fit(X_train, y_train)
        accs_train.append(forest.score(X_train, y_train))
        accs_test.append(forest.score(X_test, y_test))
        print('--- max_depth = {} ---'.format(i))
        print('Accuracy on training set: {:.3f}'.format(forest.score(X_train, y_train)))
        print('Accuracy on test set: {:.3f}'.format(forest.score(X_test, y_test)))
    # Build a simple model 0.82
    print ('Value Model: %s' % str(forest.score(X,y)) )
    #
    # Read Fasta
    for record in SeqIO.parse(fileFasta, "fasta"):
        largo=len(record)
        sequence=record.seq
        # subprocess
        peptideSmilesLi=pepLinealtoSMILE(sequence)
        peptideSmilesCy=pepCyclictoSMILE(sequence)
        molpeptideLi=Chem.MolFromSmiles(peptideSmilesLi)
        molpeptideCy=Chem.MolFromSmiles(peptideSmilesCy)
        # subprocess
        scoreSA_Li = calculateScore(molpeptideLi)
        scoreSA_Cy = calculateScore(molpeptideCy)
        # Make Prediction Random-Forest
        fp_vect_Li = genFP(molpeptideLi)
        fp_vect_Cy = genFP(molpeptideCy)
        # Get probabilities
        predictionsLi = forest.predict_proba(fp_vect_Li.reshape(1,-1))
        predictionsCy = forest.predict_proba(fp_vect_Cy.reshape(1,-1))
        #print ("Probability %s mutagenic %0.6f " % (sequence,predictionsLi[0][1]))
        #print ("Probability %s mutagenic %0.6f " % (sequence,predictionsCy[0][1]))
        # See http://cdb.ics.uci.edu/cgibin/Smi2DepictWeb.py
        QSArproperties(molpeptideLi, scoreSA_Li, sequence, predictionsLi[0][1])
        QSArproperties(molpeptideCy, scoreSA_Cy, sequence, predictionsCy[0][1])

    t2=time.time()
    resTime1=(t2-t1)
    #
    print (' Reading took %.2f seconds. \n' % resTime1 )
