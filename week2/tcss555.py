import os
import pandas as pd
import sys, getopt
import csv
import numpy as np
import sklearn
import csv
import getopt
import os
import sys
import pickle as pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

__author__ = "Pooja Shrivastava"

def main(argv):
    ipfile = ''
    opfile = ''
    try:
        opts, args = getopt.getopt(argv, 'hi:o:', ['ifile="', 'ofile='])
    except getopt.GetoptError:
        print('error:\ntcss555.py -i <inputfile> -o <outputfile>\n')
        sys.exit(2)
    for opt, arg in opts:
        if (opt == '-h'):
            print('======\nusage:\n======\ntcss555.py -i <inputfile> -o <outputfile>\n')
            sys.exit()
        elif opt in ('-i', '--ifile'):
            ipfile = arg
        elif opt in ('-o', '--ofile'):
            opfile = arg
            if not os.path.exists(opfile):
                os.makedirs(opfile)

    protest_ipfile=ipfile+'/profile/profile.csv'
    reltest_ipfile=ipfile+'/relation/relation.csv'
    # print (protest_ipfile)
    # print(reltest_ipfile)


    ############################### Gender prediction from Likes using Naive Bayes ######################################
    df_test = pd.read_csv(protest_ipfile)
    df_test_relation = pd.read_csv(reltest_ipfile)
    df_test_relation['like_id'] = df_test_relation['like_id'].astype(str)
    dfuserlike_test=df_test_relation.groupby(['userid'])['like_id'].apply(' '.join).reset_index()
    df_count = pd.merge(df_test, dfuserlike_test, on=['userid'])
    filename='/home/itadmin/ps10_code/countvectrelationlog.pk'
    vectoriser=pkl.load(open(filename,'rb'))
    reltestgender = vectoriser.transform(df_count['like_id'])
    filename= '/home/itadmin/ps10_code/NBrelationgender.pk'
    mnb=pkl.load(open(filename,'rb'))
    relgenlist=mnb.predict(reltestgender)
    relgenuseridlist = df_count['userid'].tolist()
    relgendf = pd.DataFrame(relgenlist)
    relgendf.columns=['genderrelation']
    relgenuseriddf=pd.DataFrame(relgenuseridlist)
    relgenuseriddf.columns=['userid']

    useridrelgendf = pd.concat([relgenuseriddf, relgendf], axis=1)

    genderframe = useridrelgendf
    genderlist = genderframe['genderrelation'].tolist()

    pred = []
    for i in zip(genderlist):
        pred.append(max(i, key=i.count))

    gender = pd.DataFrame(pred)
    gender.columns = ['gender']
    genderframe = pd.concat([genderframe, gender], axis=1)

    genderframe.drop(['genderrelation'], axis=1, inplace=True)

    ###########################  Generating XML  files ######################################################
    for index, row in genderframe.iterrows():
        filename =opfile
        with open(filename+row.userid+'.xml', 'w') as file:
            file.write('<user'+'\n')
            file.write('id="' + row.userid+'"\n')
            #age_groupfunc_value = age_groupfunc(row.age)
            if row.gender==0.0:
                temp="male"
            else:
                temp="female"
            #print (row.age)
            # file.write('age_group="' + row.age +'"\n')

            file.write ('age_group="xx-24"'+'\n')
            file.write('gender="' + temp + '"\n')
            file.write ('extrovert="3.49"'+ '\n')
            file.write ('neurotic="2.73"' + '\n')
            file.write ('agreeable="3.58"'+'\n')
            file.write ('conscientious="3.45"' + '\n')
            file.write ('open="3.91"'+'\n')
            file.write ('/>')

    file.close()

main(sys.argv[1:])
