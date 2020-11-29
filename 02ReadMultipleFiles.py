import os
import time
import sys
import math
import cmath
import glob
import numpy as np


# donde cada uno de los puntajes Spropiedad son funciones tipo Heavside
# que asignan 1 si el valor de tal propiedad cumple las propiedades de
# Lipinski, Ghose y Veber o 0 en caso contrario:

def Heavside (up, down, value):
    boolean = 0
    #
    if((value > down ) and (value < up)):
        boolean = 1

    return boolean

def minmax (array,title):
    max_=np.amax(array)
    min_=np.amin(array)
    desv=np.std(array)
    avg=np.average(array)
    perc=np.percentile(array, 90)
    print (title)
    percQ1=np.percentile(array, 25)
    percQ3=np.percentile(array, 75)
    print("Max: "+str(max_))
    print("Q3: "+str(percQ3))
    print("Avg: "+str(avg))
    print("Q1: "+str(percQ1))
    print("Min: "+str(min_))
    print("DS : "+str(desv))
    print("90perc: "+str(perc)+"\n")
    return avg

def sumalista(listaNumeros):
    laSuma = 0
    for i in listaNumeros:
        laSuma = laSuma + i
    return laSuma

if __name__=='__main__':
    # Time
    t1=time.time()

    totalfiles = []

    list_of_files = glob.glob('./QSAR-2D*.txt.csv')
    FO = open("01DataBase_lin_test.csv", 'w')
    FO.write("MolWeight\tMolLogP\tHbondA\tHbondD\tPolarSA\tRbonds\tMolarRefractivity\tnAtoms\tSynthAcces\tAmesMutagenic\tSeq\tSMILES\n")
    for file_name in list_of_files:
        print ("File: " + str(file_name))
        FI = open(file_name, 'r')
        for line in FI:
            line = line.replace('\n', '').replace('\r', '')
            totalfiles.append(line)
            FO.write(line + "\n")
        FI.close()
    FO.close()

    # Lists
    ListMolWeight        =[]
    ListMolLogP          =[]
    ListHbondA           =[]
    ListHbondD           =[]
    ListPolarSA          =[]
    ListRbonds           =[]
    ListMolarRefractivity=[]
    ListnAtoms           =[]
    #
    ListSynthAcces       =[]
    ListAmesMutagenic    =[]
    #
    ListScoreADME        =[]
    ListScoreTotal       =[]
    #
    ListSeq              =[]
    ##print("MolWeight\tMolLogP\tHbondA\tHbondD\tPolarSA\tRbonds\tMolarRefractivity\tnAtoms\tSynthAcces\tAmesMutagenic\n")
    for data in totalfiles:
        parameter=data.split(sep="\t",maxsplit=11)
        MolWeight        =float(parameter[0])
        MolLogP          =float(parameter[1])
        HbondA           =int(parameter[2])
        HbondD           =int(parameter[3])
        PolarSA          =float(parameter[4])
        Rbonds           =int(parameter[5])
        MolarRefractivity=float(parameter[6])
        nAtoms           =int(parameter[7])
        #
        ListMolWeight.append(MolWeight)
        ListMolLogP.append(MolLogP)
        ListHbondA.append(HbondA)
        ListHbondD.append(HbondD)
        ListPolarSA.append(PolarSA)
        ListRbonds.append(Rbonds)
        ListMolarRefractivity.append(MolarRefractivity)
        ListnAtoms.append(nAtoms)
        #
        SynthAcces       =float(parameter[8])
        AmesMutagenic    =float(parameter[9])
        ListSynthAcces.append(SynthAcces)
        ListAmesMutagenic.append(AmesMutagenic)
        #
        seq              =str(parameter[10])
        ListSeq.append(seq)


    # Promedio
    # Min Max
    minmax(ListMolWeight,"Weight")
    minmax(ListMolLogP,"LogP")
    minmax(ListHbondA,"HbondA")
    minmax(ListHbondD,"HbondD")
    minmax(ListPolarSA,"TPSA")
    minmax(ListRbonds,"Rbonds")
    minmax(ListMolarRefractivity,"MR")
    minmax(ListnAtoms,"N°")
    #
    avgSynthAcces=minmax(ListSynthAcces,"SA")
    avgAmesMutagenic=minmax(ListAmesMutagenic,"Ames")
    #
    #Proceso de testeo
    for i in range(len(ListnAtoms)):
        booleanMW  = Heavside(float(875.985) , float(160),ListMolWeight[i])
        booleanLogP= Heavside(float(5), float(-4.88),ListMolLogP[i])
        booleanHBA = Heavside(float(15), float(0),ListHbondA[i])
        booleanHBD = Heavside(float(14), float(0),ListHbondD[i])
        booleanTPSA= Heavside(float(388.029), float(160),ListPolarSA[i])
        booleanRB  = Heavside(float(27), float(0),ListRbonds[i])
        booleanMR  = Heavside(float(225), float(40),ListMolarRefractivity[i])
        booleanX   = Heavside(float(62), float(0),ListnAtoms[i])
        #
        # Propiedades fisicoquímicas ScoreADME
        sumaADME  = sumalista([booleanMW,booleanLogP,booleanHBA,booleanHBD,booleanTPSA,booleanRB,booleanMR,booleanX])
        ScoreADME = (sumaADME/8)
        #
        ListScoreADME.append(ScoreADME)
        #print (str(sumaADME) + " " + str(ScoreADME)  )

    F2 = open("ScoreTotal_lineal_test.csv", 'w')
    for i in range(len(ListScoreADME)):
        _ScoreADME       =float(ListScoreADME[i])
        #
        _SynthAcces_dM   =float(ListSynthAcces[i])
        _AmesMutagenic_dM=float(ListAmesMutagenic[i])
        avgSynthAcces    =4.626848289181031
        avgAmesMutagenic =0.3092776990414759
        #
        ScoreSA = (1 + (math.log(avgSynthAcces/_SynthAcces_dM)))
        ScoreAM = (1 + (math.log(_AmesMutagenic_dM/avgAmesMutagenic)))
        #
        ScoreTotal = (_ScoreADME + ScoreSA + ScoreAM)
        ListScoreTotal.append(ScoreTotal)
        F2.write( str(ListSeq[i])+ "\t" + str(ScoreTotal) + "\n")

    F2.close()


    print ("## Final ScoreTotal ##")
    minmax(ListScoreTotal,"Score Total")


    print ("Fin programa")
    t2=time.time()
    resTime1=(t2-t1)
    print (' Reading took %.2f seconds. \n' % resTime1 )
