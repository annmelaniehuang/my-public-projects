import pandas as pd
import datetime as dt
import re
import numpy as np
from helium import *
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait


def rego_Data_Cleaning(filepath):
    rego_data = pd.read_csv(filepath)
    TODAY_DATE = dt.datetime.today().strftime('%d-%m-%Y')

    # expiry date => datetime -> find all about to expire ones
    # high human-error identified: <- raise internal control questions
    # TODO: write handler to return permalink, entered datetime for invalid entry
    rego_data['nursing_registration_expiry_cleaned'] = \
        pd.to_datetime(rego_data['nursing_registration_expiry'], errors='coerce')  # NaT raise case
    rego_data['nurse_registration_number'] = rego_data.nurse_registration_number.str.strip()
    pattern0 = re.compile(r'[o]{3}', re.IGNORECASE)
    # TODO: write handler to return rego number in unrecognised form
    zeros_entry_incorrectly = rego_data.nurse_registration_number.str.contains(pattern0, na=False)
    rego_data['nurse_registration_number'] = \
        np.where(zeros_entry_incorrectly, rego_data.nurse_registration_number.str.replace(pattern0, '000'),
                 rego_data.nurse_registration_number)

    pattern1 = re.compile(r'([NWMnwm]{3})', re.IGNORECASE)
    pattern2 = re.compile(r'(\d{7,10})', re.IGNORECASE)
    with_header = rego_data.nurse_registration_number.str.contains(pattern1, na=False)
    rego_data['nurse_registration_number'] = \
        np.where(with_header, rego_data.nurse_registration_number.str.replace(pattern1, 'NMW'),
                 rego_data.nurse_registration_number)
    rego_data['header'] = rego_data.nurse_registration_number.str.extract(pattern1)
    rego_data['exe_digits'] = rego_data.nurse_registration_number.str.extract(pattern2)
    rego_data['rego_number_in_trackable_format'] = (rego_data.header.notnull()) & (rego_data.exe_digits.str.len() == 10)
    return rego_data


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def Convert(a_list):
    res_dct = {a_list[i]: a_list[i + 1] for i in range(0, len(a_list), 2)}
    return res_dct


def ahpra_Crawler(url, registration_number, headless_condition=True):
    driver = start_chrome('https://www.ahpra.gov.au/Registration/Registers-of-Practitioners.aspx',
                          headless=headless_condition)
    driver.implicitly_wait(3)
    write(registration_number, into='Registration number')
    press(ENTER)
    driver.implicitly_wait(3)
    results = find_all(S('.register-search-results'))
    if len(results) != 0:
        for item in results:
            results_list = item.web_element.text
            #print(results_list)
        return results_list
    else:
        return 'Not Found!'


def df_qa_report(df):
    # the function is to audit the quality of dataframe
    print('======------Data Quality Assurance------======')
    print('cleaned data frame\n\thas {0} rows and {1} columns'.format(df.shape[0], df.shape[1]))
    print('cleaning steps were performed on nursing registration expiry and registration number had string '
          'manipulation and regex pattern recognition applied')
    print('\t{0} rows will be excluded as the data is not usable even post cleaning'.
          format(df.shape[0] - df[df.rego_number_in_trackable_format].shape[0]))
    print('registration expiry that is dated after 1 June 2020 will also be excluded in this crawling session')
    print('there are {} approved nurses with registration expiry post 1/6/2020'.
          format(df[df.nursing_registration_expiry_cleaned>'2020-05-31'].shape[0]))