import os
import time
import sys
import math
import cmath
import glob
import numpy as np




if __name__=='__main__':
    # Time
    t1=time.time()

    totalScore = []
    totalSeq   = []

    list_of_files = glob.glob('./ScoreTotal*.csv')

    for file_name in list_of_files:
        print ("File: " + str(file_name))
        FI = open(file_name, 'r')
        for line in FI:
            line = line.replace('\n', '').replace('\r', '')
            parameter=line.split(sep="\t",maxsplit=3)
            mySeq    =str(parameter[0])
            MyScore  =float(parameter[1])
            #
            totalSeq.append(mySeq)
            totalScore.append(MyScore)
            #print ( str(mySeq) + " " + str(MyScore))

        FI.close()
    #
    score,seq = map(list, zip(*sorted(zip(totalScore,totalSeq),reverse=True)))
    #
    #
    FO = open("03Filter_Cyc_200.csv", 'w')
    for i in range(200):
        FO.write (str(seq[i]) + "\t" + str(score[i]) + "\n" )

    FO.close()



    print ("Fin programa")
    t2=time.time()
    resTime1=(t2-t1)
    print (' Reading took %.2f seconds. \n' % resTime1 )
