# Hospital Readmission Project - Part 1

In part 1, we perform data extraction and data preparation with a clearly defined project problem and objective.
Model selection, training and hyperparameter tuning will be performed in part 2

![](Step-by-step%20ML%20Model%20Development%20-%20Hospital%20Readmission%20%28Part%201%29_files%5Cmarkdown_1_html_image_tag_0.jpeg)

Photo by <a href="/photographer/Rotorhead-35574">Rotorhead</a> from <a href="https://freeimages.com/">FreeImages</a>

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span><ul class="toc-item"><li><span><a href="#Background" data-toc-modified-id="Background-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Background</a></span></li><li><span><a href="#About-the-Dataset" data-toc-modified-id="About-the-Dataset-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>About the Dataset</a></span></li><li><span><a href="#Objective" data-toc-modified-id="Objective-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Objective</a></span></li></ul></li><li><span><a href="#Data-Extraction" data-toc-modified-id="Data-Extraction-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Extraction</a></span><ul class="toc-item"><li><span><a href="#Read-the-documentation-(e.g.-data-dictionary)" data-toc-modified-id="Read-the-documentation-(e.g.-data-dictionary)-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Read the documentation (e.g. data dictionary)</a></span></li></ul></li><li><span><a href="#Data-Preparation" data-toc-modified-id="Data-Preparation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Preparation</a></span><ul class="toc-item"><li><span><a href="#Dropping-Columns-and-Dealing-with-Missing-Values" data-toc-modified-id="Dropping-Columns-and-Dealing-with-Missing-Values-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Dropping Columns and Dealing with Missing Values</a></span></li><li><span><a href="#Changing-Data-Types-(and-dropping-rows)" data-toc-modified-id="Changing-Data-Types-(and-dropping-rows)-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Changing Data Types (and dropping rows)</a></span></li></ul></li><li><span><a href="#Data-Visualisation" data-toc-modified-id="Data-Visualisation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Visualisation</a></span><ul class="toc-item"><li><span><a href="#Target-Balance-check" data-toc-modified-id="Target-Balance-check-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Target Balance check</a></span></li><li><span><a href="#Correlation-Heat-Map-for-All-Kept-Columns" data-toc-modified-id="Correlation-Heat-Map-for-All-Kept-Columns-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Correlation Heat Map for All Kept Columns</a></span></li><li><span><a href="#Features-Against-Target-Labels" data-toc-modified-id="Features-Against-Target-Labels-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Features Against Target Labels</a></span></li><li><span><a href="#Encoding-Features" data-toc-modified-id="Encoding-Features-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Encoding Features</a></span></li></ul></li><li><span><a href="#Store-Prepped-Data" data-toc-modified-id="Store-Prepped-Data-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Store Prepped Data</a></span></li></ul></div>


```python
#prepare environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pickle
```


```python
%config Completer.use_jedi==False
pd.set_option('display.max_columns', 500)
```

## Introduction

### Background
Succeeding a successful trial by operating a specialised unit to coordinate patients discharge. The trial not only successully reduced the number of readmission but also costs less to operate than readmission. Hospitals are considering to use patients prior data to predict risk of readmission. 

### About the Dataset

The data contains 10 years (from 1999 to 2008) of diabetes impatient encounters from 130 hospitals in the USA. The original dataset contains 69286 rows and 24 columns. Post data cleaning and transformation, he prepared dataset includes 64 feature columns. The class labels in the prepared dataset is very unbalanced with 17.2% labelled 'yes' and 82.8% labelled 'no'. Features in cleaned dataset are not scaled.

### Objective
To rollout this intervention to all patients at risk of admission, the hospital need to accurately identify these patients. This report aims to find the best performing machine learning algorithm that can help hospital to predict which patients should participate in the "discharge intervention". 

Two classification algorithms are selected for comparison and discussion: 
* **Logistic Regression Classifier**, and
* **Random Forest Classifier**. 

## Data Extraction

### Read the documentation (e.g. data dictionary)

It is important to read provided data dictionary or other relevant documentations. However, it is to be noted that different people take different approaches to help them understand the extracted dataset better. The key is to ask the right questions.

1. How big is my data set?
2. What do all the columns mean? What are the formats?
3. Is it person/transaction/group-level data?
4. How clean is the data in term of missing values and outliers?
5. How balanced is the data?
...

The raw data provided has 69286 rows and 69286 unique patient IDs. Therefore, it is patient-level data. The meaning of each column and its format are documented in data dictionary together with % of missing values.

Without reading the data documents, just by running python code to check using `isnull()`. Somewhat likely you may miss out the abnormal entries in columns. For example, the code shows that there is no missing value in the dataset. However, the data dictionary tells you the column has 97% values missing. A little bit of extra reading time saves you from risk of mistakes.

Fortunately, in this case, by printing the first 5 rows, we can identify abnormal entries in `weight` column, "?". Missing values do not always shown as Nulls or NaNs. It sometime appears in forms of odd strings, whitespaceS or punctuations.


```python
df = pd.read_csv('data/diabetes/diabetes.csv', delimiter=',')
```


```python
print('Number of unique patient IDs: {}'.format(df.patient_id.nunique()))
print('Number of rows in data: {}'.format(df.shape[0]))
if df.patient_id.nunique()==df.shape[0]:
    print('Data at patient-level.')
elif df.admission_id.nunique()==df.shape[0]:
    print('Data at admission-level.')
else: 'Further check needed'
```


```python
df.isnull().sum() # there is no nulls, contradicting with data dictionary
```


```python
df.head(5) # missing value = '?'
```

Apart from the missing values, by reading the data dictionary, you will notice there is admission type grouping provided. Such information is absolutely vital for data preparation step. Here it helps you to group all types of missing values in admission type categories together.

## Data Preparation

### Dropping Columns and Dealing with Missing Values 

`patient_id` and `admission_id` are identification numbers that are not useful, and so we will drop these two columns.

We knew from previous steps that in raw data, missing values show as '?'. Let's start by replace '?' with nulls so that we easily can size the missing value situation in each columns. `weight`, `payer_code` and `medical_specialty` columns carry too many missing values, and so they are added to the list of columns to drop.

After dropping all the columns and with missing values dealt with (dropped in this case), we can perform sanity check to confirm if all the question marks have been replace successfully (if not the code will report the column 'there were question marks removed'). At the meantime, we print number of unique values in each column. This step will help us identify which column is categorical later.


```python
#find columns with missing values and see if to drop them if meaningless
df = df.replace('?', np.nan)
nrow, ncol = df.shape
print('Data set has {0} rows and {1} columns'.format(nrow, ncol))
columns_to_drop = ['admission_id', 'patient_id']
for column in df.columns.values:
    if df[column].isnull().sum() > 0:
        columns_to_drop.append(column)
        print('{0} : {1} % missing'.format(column, df[column].isnull().sum()*100 /nrow))
```


```python
# dropping columns which are not needed
df = df.drop(columns_to_drop,axis =1)
```


```python
print('Number of unique values in each column:')
for i in df.columns.values:
    print('---{0}: {1}'.format(i, df[i].nunique()))
    if '?' in df[i].unique():
        df[i].replace('?', np.nan)
        print(i, ': there were question marks removed')
    else: 
        pass
```

### Changing Data Types (and dropping rows)

The pandas dataframe from previous step is left with on string/object and integer data. Based on knowledge from data dictionary and our unique value counts, we can confidently change below columns to categorical.This step makes column value encoding using scikit-learn much easier:
'sex','admission_type_id','discharge_disposition_id','admission_source_id','max_glu_serum', 'A1Cresult', 'group_name_1','group_name_2', 'group_name_3' and 'admission_type_grouped'

The target label column is 'readmission', which has two unique values: 'yes' and 'no'. At data preparation step, as it is unnecessary to make any change to the target, we keep it in its raw format. 

For the rest of the non-target columns, we will treat them as continuous variables and pass into `describe()` to generate basic column statistics.

For Categorical Data, if the column only contains few categories and one of them contains way too little samples comparatively, we may need to consider dropping these samples. The harm to keep is much greater than dropping. In this particular dataframe, we see `sex` column only has 3 samples of value "Unknown/Invalid".


```python
df.dtypes
```


```python
df.readmission.unique()
```


```python
# categorise data columns 
categoricals = ['sex', 'admission_type_id', 'discharge_disposition_id',
                'admission_source_id','max_glu_serum', 'A1Cresult', 'group_name_1', 
                'group_name_2', 'group_name_3']
df[categoricals] = df[categoricals].astype('str').astype('category')
```


```python
for i in categoricals:
    print(i)
    print(df[i].value_counts().sort_index())
```


```python
df = df[df.sex != 'Unknown/Invalid'] # drop rows for unknown/invalid value in sex column
```


```python
# cleaning admission type column
admission_type_gp_map = {
    '1' : 'emergency',
    '2' : 'urgent',
    '3' : 'elective',
    '4' : 'newborn',
    '5' : 'not avaialble/null',
    '6' : 'not avaialble/null',
    '7' : 'trauma centre',
    '8' : 'not avaialble/null',
}

df['admission_type_grouped'] = df.admission_type_id.astype(str).replace(admission_type_gp_map)
```


```python
# Mapping admission_source_id to discharge_disposition_grouped
dict_map = ({1: 'Discharged to home',
             3: 'Transferred to SNF',
             6: 'Home health service',
             18: 'Other', 
             2: 'Short term hospital', 
             22: 'Short term hospital',
             5: 'Short term hospital',
             25: 'Other',
             4: 'Other', 
             7: 'Other',
             23: 'Other',
             28: 'Other', 
             8: 'Other',
             15: 'Other', 
             24: 'Other', 
             9: 'Other', 
             17: 'Other', 
             10: 'Other',
             27: 'Other',
             16: 'Other',
             12: 'Other'})
df['discharge_disposition_grouped'] = df.discharge_disposition_id.astype(str).replace(dict_map)
```


```python
dict_map = ({7: 'Emergency Room', 
             1: 'Physician Referral',
             4: 'Transfer from another health care facility',
             6: 'Transfer from another health care facility',
             2: 'Other',
             3: 'Other',
             5: 'Other', 
             8: 'Other',
             9: 'Other', 
             10: 'Other',
             11: 'Other',
             12: 'Other', 
             13: 'Other',
             14: 'Other', 
             15: 'Other',  
             17: 'Other', 
             18: 'Other',
             19: 'Other',
             20: 'Other',
             21: 'Other',
             22: 'Other',
             23: 'Other',
             24: 'Other',
             25: 'Other',
             26: 'Other'})
df['admission_source_grouped'] = df.admission_source_id.astype(str).replace(dict_map)
```


```python
df = df.drop(['discharge_disposition_id','admission_source_id','admission_type_id'], axis = 1)
```


```python
# categorise data columns 
categoricals = ['sex', 'admission_type_grouped', 'discharge_disposition_grouped',
                'admission_source_grouped','max_glu_serum', 'A1Cresult', 'group_name_1', 
                'group_name_2', 'group_name_3']
df[categoricals] = df[categoricals].astype('str').astype('category')
```


```python
df.dtypes
```


```python
CONTINUOUS_VARIABLES = [x for x in df.columns 
                        if x not in categoricals and x != 'readmission']
```


```python
df.describe()
```

## Data Visualisation

### Target Balance check

Time to check the target label balance. Noticed that the entry is yes and no. For ease of training later, label is transformed into binary: yes = 1, no = 0.

83% were negative case and 17% were positive case. The dataset is clearly imbalaced.


```python
print(df.readmission.value_counts())
print(df.readmission.value_counts(normalize=True))
```

### Correlation Heat Map for All Kept Columns

How do use pair-wise correlation plot to help better understand the features we are given? What does it mean that two features are highly correlated? What do we do?

Nothing to fear if you see pairs of your features are highly correlated at this stage. All you need to do is bear that in mind, because there are only two ways around the matter:

1. remove them when chosen method/algorithm cannot cope with multicolinearity
2. keep them but use method/algorithm that is immue to multicolinearity

It is a key step for you to justify your later choice of a model or extra steps of feature removal/engeering. Not just a fancy colourful plot to show you have done the step.

Here, I also add an additional step to plot the correlation (Pearson's) for each target label to double check if the pattern differs. It appears that for the 'yes' label, the pair-wise correlation of `los` and `number_of_diagnoses` are higher.

I know I mentioned earlier that there is no need to panik when there are pairs of highly-correlated feature columns. But when should you be alarmed? Answer is casaulity. If two features are **KNOWN** to have causality relationship, you must remove one of them. What does it mean by known? This often comes by design or domain knowledge, one is caused by another, e.g. completion of the form is caused by completing form section 1 and 2. Causality is extremely to prove in reality, and it must have been engraved in everyone's head that correlation does not mean causality. Therefore, make sure to only be alarmed and take actions when there is **CONFIDENTLY KNOWN CASUALITY**.


```python
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df.drop(['readmission'], axis=1).corr(), annot=True, cmap='YlOrRd', ax=ax)
```


```python
corr_matrix = df.corr()
abs_corr_matrix = corr_matrix.abs()
```


```python
abs_corr_matrix_unstacked = abs_corr_matrix.unstack().sort_values(ascending=False)
print('Pairs of features with the strongest correlation')
abs_corr_matrix_unstacked[abs_corr_matrix_unstacked<1.0][1:20]
```


```python
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df.loc[df.readmission=='yes'].drop(['readmission'], axis=1).corr(), annot=True, cmap='YlOrRd', ax=ax)
```


```python
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df.loc[df.readmission=='no'].drop(['readmission'], axis=1).corr(), annot=True, cmap='YlOrRd', ax=ax)
```

### Features Against Target Labels

For continuous variables, data may distribute differently by target labels. This step is particilarly achieving and helpful. You can see the measures differ between the two classes ('yes' and 'no' in this case). Simply inspecting the quartiles, medians, means and standard deviations gives you valuable information such as the readmission group are generally older than the non-readmission group. Results produced from this step can later feed into feature exploration and engineering to achieve better model performance. 

Here, I used `seaborn` to plot the columns and colour by target label of readmission. But there are other libraries allow you to achieve the same results. The judgement here is really what type of visualisation you choose to tell what story. 


```python
# to use seaborn to plot, prep the dataframe to long form
# readmission label -> each feature -> value
# index, var_name, value_name
df_for_plot = df.melt(['readmission']
                      , var_name = 'cols'
                      , value_name = 'vals')
```


```python
# Admited cases <- continuous variable
df.loc[df.readmission == 'yes', df.columns != 'readmission'].describe()
```


```python
# Not readmitted <- continuous variable
df.loc[df.readmission == 'no', df.columns != 'readmission'].describe()
```


```python
y = df['readmission']
continuous_df = pd.concat([df[CONTINUOUS_VARIABLES], y], axis = 1)
categorical_df = pd.concat([df[categoricals],y], axis = 1)
```


```python
continous_df_melt = pd.melt(continuous_df
                            , id_vars=['readmission']
                            , var_name = 'cols'
                            , value_name= 'vals')
categorical_df_melt = pd.melt(categorical_df
                            , id_vars=['readmission']
                            , var_name = 'cols'
                            , value_name= 'vals')
```


```python
p = sns.FacetGrid(continous_df_melt
                  , col = 'cols'
                  , hue = 'readmission'
                  , col_wrap= 3
                  , sharex=False
                  , sharey=False
                  , margin_titles=True
                  , legend_out=True
    )
pg = (p.map(sns.kdeplot, "vals", alpha=0.6, shade=True, bw=1.5).add_legend())
#pg = p.map(plt.hist, 'vals', alpha = 0.6).add_legend()
pg.fig.suptitle('Continuous Variable Histograms - Colour by Label', size=16)
pg.fig.subplots_adjust(top=.9)
pg.fig.set_size_inches(14,13)
for ax in pg.axes.ravel():
    ax.legend(title="Readmission"
             , loc='upper right')
```


```python
continous_df_melt['readmission'] = continous_df_melt.readmission.astype('str')
```


```python
p = sns.axisgrid.FacetGrid(continous_df_melt
                           , col = 'cols'
                           , sharex = True
                           , sharey = False
                           , margin_titles = True
                           , col_wrap= 4
)
pg = p.map(sns.boxplot, 'readmission', 'vals'
            , order = ['no', 'yes']
            , palette= ['orange', 'blue'], notch=True).add_legend()
pg.fig.suptitle('Continuous Variables Boxplots - Colour by Label', size=16)
pg.fig.subplots_adjust(top=.9)
pg.fig.set_size_inches(14,13)
```


```python
p = sns.FacetGrid(continous_df_melt
                  , col = 'cols'
                  , hue = 'readmission'
                  , col_wrap= 3
                  , sharex=False
                  , sharey=False
                  , margin_titles=True
                  , legend_out=True
    )
pg = p.map(plt.hist
           , 'vals'
           , alpha = 0.6
           , density=True).add_legend()
pg.fig.suptitle('Continuous Variable Histograms - Colour by Label', size=16)
pg.fig.subplots_adjust(top=.9)
pg.fig.set_size_inches(14,13)
for ax in pg.axes.ravel():
    ax.legend(title="Readmission"
             , loc='upper right')
```


```python
p = sns.FacetGrid(continous_df_melt
                           #, row = 'cols'
                           , col = 'cols'
                           , sharex = True
                           , sharey = False
                           , margin_titles = True
                           , col_wrap= 3
)
pg = p.map(sns.violinplot, 'readmission', 'vals' 
            #, order = ['0', '1']
            , palette= ['orange', 'blue']
          ).add_legend()
pg.fig.suptitle('Continuous Variables Violinplots - Colour by Label', size=16)
pg.fig.subplots_adjust(top=.9)
pg.fig.set_size_inches(14,13)
```


```python
p = sns.PairGrid(continuous_df
                 #, row = 'cols'
                 #, col = 'cols'
                 , hue = 'readmission'
                 #, sharex = True
                 #, sharey = False
                 #, margin_titles = True
                 #, col_wrap= 3
)
pg = p.map(plt.scatter)
#pg = p.map_diag(plt.hist, alpha = 0.8)
#pg = pg.map_offdiag(plt.scatter)
#pg = pg.add_legend()
pg.fig.suptitle('Continuous Variables Pair Scatter - Colour by Label', size=16)
pg.fig.subplots_adjust(top=.9)
pg.fig.set_size_inches(14,13)
```


```python
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(categoricals, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
```

### Encoding Features

Again, different people take different approaches to end data preparation step. Some may question, "wait, did you forget to encode your categorical features?"

The answer is no. Feature encoding, engineering and etc are more advanced steps. There are a few encoding methods. When and how to encode should be carely assessed, considering factors such as how often values in features change and how often the model is retrained and etc. 

Here I decide that one-hot-encoder is good enough for this particular case. Therefore, `get_dummies()` from pandas libray is applied to all categorical columns.


```python
hospital = pd.get_dummies(df.drop(['readmission'], axis=1))
```

## Store Prepped Data

Congratulations! You have now completed the data extraction and preparation. Using pickly to dump the data in you local repo so it is ready for later steps down the ML model delopment pipeline (see Part 2).


```python
import pickle

with open('hospital_data.pickle', 'wb') as output:
    pickle.dump(hospital, output)
```
