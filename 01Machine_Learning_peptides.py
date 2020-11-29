import os
import time
import sys
import math
import gzip
import pickle
import glob
import numpy as np
#
from multiprocessing import Process
from joblib import Parallel, delayed
import multiprocessing
#
from multiprocessing.dummy import Pool as ThreadPool
#
from collections import defaultdict
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
    ##Ertl, P. and Schuffenhauer A. “Estimation of Synthetic Accessibility
    ##Score of Drug-like Molecules based on Molecular Complexity and Fragment
    ##Contributions” Journal of Cheminformatics 1:8 (2009)
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

def QSArproperties_test(array,forest, num, namefile):
    ##Bickerton, G.R.; Paolini, G.V.; Besnard, J.; Muresan, S.; Hopkins, A.L. (2012)
    ##Quantifying the chemical beauty of drugs,
    ##Nature Chemistry, 4, 90-98
    ##[https://doi.org/10.1038/nchem.1243]
    #
    ##Wildman, S.A.; Crippen, G.M. (1999)
    ##Prediction of Physicochemical Parameters by Atomic Contributions,
    ##Journal of Chemical Information and Computer Sciences, 39, 868-873
    ##[https://doi.org/10.1021/ci990307l]
    #
    fw = open( 'QSAR-2D' + str(num) + str(namefile) + '.csv', 'w')
    #
    for line in array:
        parameter= line.split(sep="\t",maxsplit=9)
        peptide_seq = parameter[0]
        peptide     = parameter[1]
        #
        molpeptideLi  = Chem.MolFromSmiles(peptide)
        # subprocess
        scoreSA_Li    = calculateScore(molpeptideLi)
        #
        # Make Prediction Random-Forest
        fp_vect_Li    = genFP(molpeptideLi)
        # Get probabilities
        predictionsLi = forest.predict_proba(fp_vect_Li.reshape(1,-1))
        #print ("Probability %s mutagenic %0.6f " % (sequence,predictionsLi[0][1]))
        # See http://cdb.ics.uci.edu/cgibin/Smi2DepictWeb.py
        propQSAR      = QED.properties(molpeptideLi)
        MolWeight     = propQSAR.MW
        MolLogP       = propQSAR.ALOGP
        HbondA        = propQSAR.HBA
        HbondD        = propQSAR.HBD
        PolarSA       = propQSAR.PSA
        Rbonds        = propQSAR.ROTB
        Aromatic      = propQSAR.AROM
        #
        MolarRefractivity = Crippen.MolMR(molpeptideLi)
        nAtoms            = molpeptideLi.GetNumAtoms()
        #
        SynthAcces    = scoreSA_Li
        AmesMutagenic = predictionsLi[0][1]
        #
        result = ( str(MolWeight) + "\t" + str(MolLogP) + "\t" + str(HbondA) + "\t" + str(HbondD) + "\t" + str(PolarSA) + "\t" + str(Rbonds) + "\t" + str(MolarRefractivity) + "\t" + str(nAtoms) + "\t" + str(SynthAcces) + "\t" + str(AmesMutagenic) + "\t" + str (peptide_seq) + "\t" + str(peptide) + "\n")
        #print (result)
        fw.write(result)

    fw.close()



if __name__=='__main__':
    #
    # Time
    t1=time.time()
    # Data Base Synthetic Accessibility
    readFragmentScores("fpscores")
    ##Hansen, K., Mika, S., Schroeter, T., Sutter, A., ter Laak, A.,
    ##Steger-Hartmann, T., Heinrich, N., and Muller, K. R. (2009)
    ##Benchmark Data Set for in Silico Prediction of Ames Mutagenicity.
    ##J. Chem. Inf. Model. 49, 2077−2081.
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
##    for i in [1,1000]:
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
    #
    outputBase = 'output' # output.1.txt, output.2.txt, etc.

    path = 'Values*'
    files = glob.glob(path)
    for name in files:
        # This is shorthand and not friendly with memory
        # on very large files (Sean Cavanagh), but it works.
        input     = open(name, 'r').read().split('\n')
        inputread = open(name, 'r')
        splitLen  = len(inputread.readlines())
        inputread.close()
        div_lines = int ((splitLen / 20) + 2)
        print ("Total lines :" + str(splitLen))
        print ("Div lines   :" + str(div_lines))

        at = 1
        for lines in range(0, len(input), div_lines):
            # First, get the list slice
            outputData = input[lines:lines+div_lines]
            # Now open the output file, join the new slice with newlines
            # and write it out. Then close the file.
            output = open(outputBase + str(at) + '.txt', 'w')
            output.write('\n'.join(outputData))
            output.close()
            # Increment the counter
            at += 1

        print ("\nFinal dividir archivo principal\n")

        # Read Smiles
        smiles_1 = []
        smiles_2 = []
        smiles_3 = []
        smiles_4 = []
        smiles_5 = []
        smiles_6 = []
        smiles_7 = []
        smiles_8 = []
        smiles_9 = []
        smiles_10 = []
        smiles_11 = []
        smiles_12 = []
        smiles_13 = []
        smiles_14 = []
        smiles_15 = []
        smiles_16 = []
        smiles_17 = []
        smiles_18 = []
        smiles_19 = []
        smiles_20 = []
        #
        filename_1 = "output1.txt"
        f_1=open(filename_1, "r")
        f1_1 = f_1.readlines()
        f_1.close()
        for x in f1_1:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_1.append(x)

        filename_2 = "output2.txt"
        f_2=open(filename_2, "r")
        f1_2 = f_2.readlines()
        f_2.close()
        for x in f1_2:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_2.append(x)

        filename_3 = "output3.txt"
        f_3=open(filename_3, "r")
        f1_3 = f_3.readlines()
        f_3.close()
        for x in f1_3:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_3.append(x)

        filename_4 = "output4.txt"
        f_4=open(filename_4, "r")
        f1_4 = f_4.readlines()
        f_4.close()
        for x in f1_4:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_4.append(x)

        filename_5 = "output5.txt"
        f_5=open(filename_5, "r")
        f1_5 = f_5.readlines()
        f_5.close()
        for x in f1_5:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_5.append(x)

        filename_6 = "output6.txt"
        f_6=open(filename_6, "r")
        f1_6 = f_6.readlines()
        f_6.close()
        for x in f1_6:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_6.append(x)

        filename_7 = "output7.txt"
        f_7=open(filename_7, "r")
        f1_7 = f_7.readlines()
        f_7.close()
        for x in f1_7:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_7.append(x)

        filename_8 = "output8.txt"
        f_8=open(filename_8, "r")
        f1_8 = f_8.readlines()
        f_8.close()
        for x in f1_8:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_8.append(x)

        filename_9 = "output9.txt"
        f_9=open(filename_9, "r")
        f1_9 = f_9.readlines()
        f_9.close()
        for x in f1_9:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_9.append(x)

        filename_10 = "output10.txt"
        f_10=open(filename_10, "r")
        f1_10 = f_10.readlines()
        f_10.close()
        for x in f1_10:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_10.append(x)

        #
        #
        #
        filename_11 = "output11.txt"
        f_11=open(filename_11, "r")
        f1_11 = f_11.readlines()
        f_11.close()
        for x in f1_11:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_11.append(x)

        filename_12 = "output12.txt"
        f_12=open(filename_12, "r")
        f1_12 = f_12.readlines()
        f_12.close()
        for x in f1_12:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_12.append(x)

        filename_13 = "output13.txt"
        f_13=open(filename_13, "r")
        f1_13 = f_13.readlines()
        f_13.close()
        for x in f1_13:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_13.append(x)

        filename_14 = "output14.txt"
        f_14=open(filename_14, "r")
        f1_14 = f_14.readlines()
        f_14.close()
        for x in f1_14:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_14.append(x)

        filename_15 = "output15.txt"
        f_15=open(filename_15, "r")
        f1_15 = f_15.readlines()
        f_15.close()
        for x in f1_15:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_15.append(x)

        filename_16 = "output16.txt"
        f_16=open(filename_16, "r")
        f1_16 = f_16.readlines()
        f_16.close()
        for x in f1_16:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_16.append(x)

        filename_17 = "output17.txt"
        f_17=open(filename_17, "r")
        f1_17 = f_17.readlines()
        f_17.close()
        for x in f1_17:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_17.append(x)

        filename_18 = "output18.txt"
        f_18=open(filename_18, "r")
        f1_18 = f_18.readlines()
        f_18.close()
        for x in f1_18:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_18.append(x)

        filename_19 = "output19.txt"
        f_19=open(filename_19, "r")
        f1_19 = f_19.readlines()
        f_19.close()
        for x in f1_19:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_19.append(x)

        filename_20 = "output20.txt"
        f_20=open(filename_20, "r")
        f1_20 = f_20.readlines()
        f_20.close()
        for x in f1_20:
            # delete \n
            x = x.replace('\n', '').replace('\r', '')
            smiles_20.append(x)


        p1 = Process(target=QSArproperties_test, args=(smiles_1,forest,1,name))
        p2 = Process(target=QSArproperties_test, args=(smiles_2,forest,2,name))
        p3 = Process(target=QSArproperties_test, args=(smiles_3,forest,3,name))
        p4 = Process(target=QSArproperties_test, args=(smiles_4,forest,4,name))
        p5 = Process(target=QSArproperties_test, args=(smiles_5,forest,5,name))
        p6 = Process(target=QSArproperties_test, args=(smiles_6,forest,6,name))
        p7 = Process(target=QSArproperties_test, args=(smiles_7,forest,7,name))
        p8 = Process(target=QSArproperties_test, args=(smiles_8,forest,8,name))
        p9 = Process(target=QSArproperties_test, args=(smiles_9,forest,9,name))
        p10 = Process(target=QSArproperties_test, args=(smiles_10,forest,10,name))
        p11 = Process(target=QSArproperties_test, args=(smiles_11,forest,11,name))
        p12 = Process(target=QSArproperties_test, args=(smiles_12,forest,12,name))
        p13 = Process(target=QSArproperties_test, args=(smiles_13,forest,13,name))
        p14 = Process(target=QSArproperties_test, args=(smiles_14,forest,14,name))
        p15 = Process(target=QSArproperties_test, args=(smiles_15,forest,15,name))
        p16 = Process(target=QSArproperties_test, args=(smiles_16,forest,16,name))
        p17 = Process(target=QSArproperties_test, args=(smiles_17,forest,17,name))
        p18 = Process(target=QSArproperties_test, args=(smiles_18,forest,18,name))
        p19 = Process(target=QSArproperties_test, args=(smiles_19,forest,19,name))
        p20 = Process(target=QSArproperties_test, args=(smiles_20,forest,20,name))

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
        p11.start()
        p12.start()
        p13.start()
        p14.start()
        p15.start()
        p16.start()
        p17.start()
        p18.start()
        p19.start()
        p20.start()

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
        p11.join()
        p12.join()
        p13.join()
        p14.join()
        p15.join()
        p16.join()
        p17.join()
        p18.join()
        p19.join()
        p20.join()

        print ("Ciclos : " + str(name))


    print ("Fin programa")
    t2=time.time()
    resTime1=(t2-t1)
    #
    print (' Reading took %.2f seconds. \n' % resTime1 )
