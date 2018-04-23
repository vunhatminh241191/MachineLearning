import csv
import getopt
import os
import sys
from typing import List

import numpy as np

__author__ = "Pooja Shrivastava"


def age_groupfunc(age_value):
    if age_value<=24:
        return str("xx-24")
    elif age_value<=34:
        return str("25-34")
    elif age_value<=49:
        return str("35-49")
    else:
        return str("50-xx")



def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,'hi:o:',['ifile="','ofile='])
    except getopt.GetoptError:
        print ('error:\ntcss555.py -i <inputfile> -o <outputfile>\n')
        sys.exit(2)
    for opt, arg in opts:
        if (opt == '-h'):
            print ('======\nusage:\n======\ntcss555.py -i <inputfile> -o <outputfile>\n')
            sys.exit()
        elif opt in ('-i', '--ifile'):
            inputfile = arg
        elif opt in ('-o', '--ofile'):
            opfile = arg
            if not os.path.exists(opfile):
                os.makedirs(opfile)

    profile_ip = inputfile + 'profile/profile.csv'
    f=open("C:/temp/tcss555/training/profile/profile.csv")
    r_csv = csv.reader(f)
    next(r_csv)
    count = 0
    gender = 0
    opn = 0
    con = 0
    ext = 0
    agr = 0
    neu = 0
    age_group: List[int] = [0, 0, 0, 0]

    for row in r_csv:
        count += 1
        gender += int(float(row[3]))
        opn += float(row[4])
        con += float(row[5])
        ext += float(row[6])
        agr += float(row[7])
        neu += float(row[8])

        a = int(row[2])
        if a <= 24:
            age_group[0] += 1
        elif a <= 34:
            age_group[1] += 1
        elif a <= 49:
            age_group[2] += 1
        elif a >50:
            age_group[3] += 1

        filename ="C:/Users/Pooja/Desktop/XML/"
        with open(filename+row[1]+'.xml', 'w') as file:

            file.write('<user' + '\n')
            file.write('id="' + row[1] + '"\n')
            age_groupfunc_value = age_groupfunc(float(row[2]))
            if row[3] == 0.0:
               temp = "male"
            else:
               temp = "female"

               file.write ('age_group="' + age_groupfunc_value + '"\n')
               file.write ('gender="' + temp + '"\n')
               file.write ('extrovert="' + str(round((float(row[6])), 2)) + '"\n')
               file.write ('neurotic="' + str(round(float((row[8])), 2)) + '"\n')
               file.write ('agreeable="' + str(round(float((row[7])), 2)) + '"\n')
               file.write ('conscientious="' + str(round(float((row[5])), 2)) + '"\n')
               file.write ('open="' + str(round((float(row[4])), 2)) + '"\n')
               file.write ('/>')
    x = (int(gender / (count / 2)))
    v = np.argmax(age_group)

    if (v ==0):
        agrp =str("xx-24")
    elif (v ==1):
        agrp=str("25-34")
    elif (v ==2):
        agrp= str("35-49")
    else:
        agrp= str("50-xx")

    print('{} {} {} {} {} {} {}'.format(x, opn/count, con/count, ext/count, agr/count, neu/count, agrp))
    print('I am here again')

    f.close()


main(sys.argv[1:])