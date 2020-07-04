from helium import *
from selenium.webdriver.common.keys import Keys
import pandas as pd
from services import *
import re
import pause
import time
from numpy.random import seed
from numpy.random import randint
import timeit

seed(1)
FILE = 'D:\\approved_nurses.csv'
URL = 'https://www.ahpra.gov.au/Registration/Registers-of-Practitioners.aspx'


df = rego_Data_Cleaning(FILE)
df_cleaned = df[df.rego_number_in_trackable_format]
df_qa_report(df)
good_rego_number = df_cleaned[~(df_cleaned.nursing_registration_expiry_cleaned > '2020-05-31')]
list_good_rego_number = good_rego_number.header.str.cat(good_rego_number.exe_digits,
                                                        sep='', join='left').values  # to array

# divide list into a list of 90 small lists
list_of_list = list(divide_chunks(list_good_rego_number, 3)) #list of lists of threes [[x,y,z],[x,y,z],[x,y,z], ...]
# divide 90 small lists into chunks of 20 of 3 small lists
test = list(divide_chunks(list_of_list, 20))
# devide list of lists into 5 chunks [[x,y,z],[x,y,z],[x,y,z]...size=30],[[x,y,z],[x,y,z],[x,y,z],...size=30],...]

# def main(url = URL, data=test):
chunk_concat = pd.DataFrame()
for j, chunk in enumerate(test):
    print('chunk number {} starts: '.format(j))
    concat_list = []
    empty_df = pd.DataFrame()
    for ith_list in chunk:
        sub_list = []
        for i, val in enumerate(ith_list):
            start = time.time()
            results = ahpra_Crawler(url=URL, registration_number=val, headless_condition=True)
            if results == 'Not Found!':
                pass
            else:
                try:
                    results_list = results.replace(':', '').split('\n')
                    results_list[7] = ', '.join(results_list[7:9])
                    results_list.pop(8)
                    results_list.pop(-1)
                    title = results_list[0].split(' ')
                    profession = results_list[1].split(' ')
                    professional = Convert(results_list[2:])
                    professional['Prefix'] = title[0]
                    professional['Last_name'] = title[-1]
                    professional['Given_name'] = ' '.join(title[1:-1])
                    professional['Profession'] = profession[1]
                    sub_list.append(professional)
                    stop = time.time()
                    print('finished in {}s \n'.format(stop-start))
                    # print('pause {0}'.format(time.ctime()))
                    # print('pause down {0}'.format(time.ctime()))
                except IndexError:
                    pass
                time_space_rand = randint(5, 10, 1)[0]
                #print('Auto pausing of {}s is randomly chosen-----\n'.format(time_space_rand))
                pause.seconds(time_space_rand)
                d = pd.DataFrame(sub_list)
                empty_df = empty_df.append(d)
                concat_list.append(sub_list)
            chunk_concat = chunk_concat.append(empty_df)
            chunk_concat_new = chunk_concat.drop_duplicates()
            file_name = 'go_'+'1'+'.csv'
            chunk_concat_new.to_csv(file_name)
            print('chunk.csv dumped')
            #print('auto pausing triggered again-------\n')
            pause.seconds(randint(7, 10, 1)[0])
