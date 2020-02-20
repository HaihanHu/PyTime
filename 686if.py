#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:33:37 2019

@author: shirleyhu
"""

import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/shirleyhu/Documents/686/686newdata.csv', low_memory=False, encoding='latin-1')
df.PY.value_counts()
pd.DataFrame(df.SO.value_counts())

ifs = {i: pd.read_csv('/Users/shirleyhu/Documents/686/final/JournalHomeGrid{}.csv'.format(i)) for i in range(2000,2019)}

def ifpreprocess(df):
    df['Full Journal Title'] = df['Full Journal Title'].str.upper()
    df.drop_duplicates(subset=['Full Journal Title'], keep='first', inplace=True)      
    return df.set_index('Full Journal Title')['Journal Impact Factor']

for i in ifs:
    ifs[i]=ifpreprocess(ifs[i])

conditions=[]
for n in range(2000,2019):
    conditions.append((df['PY']==n))
   
choices = []
for n in range(2000,2019):
    choices.append((df['SO'].map(ifs[n])))
    
df['IM'] = np.select(conditions,choices)
df['IM'].fillna(value=0,inplace=True)
df['IM'].replace("Not Available", 0, inplace=True)
df.to_csv('/Users/shirleyhu/Documents/686/686newdatawithim.csv')

##calculate authors impact factor

df_im = pd.read_csv('/Users/shirleyhu/Documents/686/686newdatawithim.csv',low_memory=False)
df_im['fund_label'] = df_im['FU'].fillna(0).apply(lambda x: 1 if x!=0 else 0)

## Impact factor

df_im['IM'] = pd.to_numeric(df_im['IM'])
dfa = df_im.AF.str.split(';', expand=True).\
                   join(df_im.fund_label).join(df_im.PY).join(df_im['IM'])
dfa.fillna(value=np.nan, inplace=True)

def author_if(df):
    temp=[]
    temp_n=[]
    for row in df.iterrows():
        index, data = row
        temp.append(data.tolist())
    for i in range(len(temp)):
        for j in range(len(temp[i])-3):
            temp_n.append([temp[i][j],temp[i][-3],temp[i][-2],temp[i][-1]])
    df_n = pd.DataFrame(temp_n, columns=['author','fund','PY','IM']).dropna()
    return df_n

df_authors=author_if(dfa)
df_authors['author'] = df_authors['author'].str.lstrip()
dfa_if = pd.pivot_table(df_authors, index=['author', 'PY'], values='IM',aggfunc=np.sum)
dfa_if_unstacked = dfa_if.unstack()
dfa_if_unstacked['sum']=dfa_if_unstacked.sum(axis = 1, skipna = True)
dfa_if_unstacked.columns = dfa_if_unstacked.columns.droplevel(0)
dfa_if_unstacked=dfa_if_unstacked.reset_index()
dfa_if_unstacked.columns=['author','2000if','2001if','2002if','2003if','2004if','2005if','2006if','2007if','2008if','2009if','2010if','2011if','2012if','2013if','2014if','2015if','2016if','2017if','2018if','sum']
dfa_if_unstacked.to_csv('/Users/shirleyhu/Documents/686/686withim.csv')

## funding percentage

dfa_fund = pd.pivot_table(df_authors, values='fund', index=['author', 'PY'], aggfunc=np.sum)
dfa_fund_unstacked = dfa_fund.unstack()
dfa_fund_unstacked['sum']=dfa_fund_unstacked.sum(axis = 1, skipna = True)
dfa_fund_unstacked['sum_01']=(dfa_fund_unstacked ==0).astype(int).sum(axis = 1) + dfa_fund_unstacked['sum']
dfa_fund_unstacked['fund_perc'] = dfa_fund_unstacked['sum']/dfa_fund_unstacked['sum_01']
dfa_fund_unstacked.to_csv('/Users/shirleyhu/Documents/686/686withfund.csv')

## h-index
h_index = pd.read_csv('/Users/shirleyhu/Documents/686/hindex.csv')
df= pd.read_csv('/Users/shirleyhu/Documents/686/df.csv')

df_merge_withh = pd.merge(df, h_index, how = 'inner')
df_merge_withh.to_csv('/Users/shirleyhu/Documents/686/686withh.csv')
