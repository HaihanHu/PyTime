#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:50:18 2019

@author: shirleyhu
"""
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np

data_n= pd.read_csv("/Users/shirleyhu/Documents/686/686finaldata.csv")

##plot
fig, ax = plt.subplots(figsize=(20,7))
ax=sns.countplot(x = "PY", data = data_n,\
             palette='Blues_d',alpha=0.7)
sns.despine()
ax.spines['bottom'].set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(20,10))
ax=sns.countplot(y = "PY", data = data_n,\
             order = data_n['PY'].value_counts().index, palette='Blues_d',alpha=0.7)
sns.despine()
ax.spines['bottom'].set_visible(False)
##for p in ax.patches:
##    ax.annotate('{:}'.format(p.get_width()), (p.get_width()+20, p.get_y()),va="top")
##plt.title("Product Distribution")
##plt.savefig("product.jpg", bbox_inches="tight")
#plt.show()



##extract authors
dn_author=data_n.AF.str.split(';', expand=True).add_prefix('au')
dn_author = dn_author.applymap(lambda x: x.lstrip() if type(x) is str else x)
dn_author.fillna(value=np.nan, inplace=True)

dn_author.to_csv(r'/Users/shirleyhu/Documents/686/authornet.txt', header=None, index=None, sep=';',mode='a')
G = nx.read_adjlist('/Users/shirleyhu/Documents/686/authornet.txt',delimiter=';',create_using=nx.DiGraph)
G.remove_node('')
G.number_of_nodes(),G.number_of_edges()
df_edges = pd.DataFrame(list(G.edges), columns=['author1', 'author2'])

## centrality

closeCent = nx.closeness_centrality(G)
closesort=sorted(closeCent.items(),key=operator.itemgetter(1),reverse=True)
df_close = pd.DataFrame(closesort)
degree = nx.degree_centrality(G)
#between = nx.betweenness_centrality(G)
lst=[]
for n in G:
    lst.append([n,degree[n]])
df_nodes = pd.DataFrame(lst)
df_nodes.columns=['author','degree']

dfa_if_unstacked= pd.read_csv('/Users/shirleyhu/Documents/686/686withim.csv')

df_merge = pd.merge(dfa_if_unstacked, df_nodes, how = 'inner')
df_merge.to_csv('/Users/shirleyhu/Documents/686/686withdegree.csv')
