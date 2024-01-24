#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
get_ipython().run_line_magic('matplotlib', 'inline')

from minisom import MiniSom
from sklearn.preprocessing import minmax_scale, scale


# In[88]:


get_ipython().system('pip list')


# In[2]:


#now load in my data 
import os.path

data = pd.read_csv('C:/Users/ack98/Downloads/hou_likeness/UpopRes1_19.csv')
data.head()


# In[36]:


#make a color for each UrbanPop cluster and each MAxEnt label. This will help visualize results 
cluster_color = {1: 'darkgreen', 2: 'limegreen',3: 'darkorange',4: 'crimson',5: 'brown',6:'midnightblue',7: 'fuchsia',
                 8: 'black'}

maxlabel_color = {'Very High': 'black', 'High': 'darkorange', 'Medium': 'brown','Low': 'limegreen','Very Low': 'darkgreen'}

colors_dict = {c: maxlabel_color[dm] for c, dm in zip(data.GEOID,
                                                      data.MaxLabel)}


# In[83]:


#now isolate the features 
feature_names = ['IndLiving', 'Poverty', 'CommUnemp',
                 'SCR', 'MEAN']


# In[84]:


#now scale the features 
X = data[feature_names].values
X = scale(X)
#print(X)
#len(X)


# In[85]:


#now train the SOM
size = 15
som = MiniSom(size, size, len(X[0]),
              neighborhood_function='gaussian', sigma=1.5,
              random_seed=1)

som.pca_weights_init(X)
som.train_random(X, 1000, verbose=True)


# In[53]:


get_ipython().system('pip install plotly')


# In[86]:


#create property plot to show mean values of each neuron 
#The property plot shows the mean values of each neuron seperately plotted for each property. 
#This view can be used to explore the "correlation" of properties.

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
def showPropertyPlot(som, data, columns):
# plots the distances for each different property
    win_map = som.win_map(data)
    size=som.distance_map().shape[0]
    properties=np.empty((size*size,2+data.shape[1]))
    properties[:]=np.NaN
    i=0
    for row in range(0,size):
        for col in range(0,size):
            properties[size*row+col,0]=row
            properties[size*row+col,1]=col

    for position, values in win_map.items():
        properties[size*position[0]+position[1],0]=position[0]
        properties[size*position[0]+position[1],1]=position[1]
        properties[size*position[0]+position[1],2:] = np.mean(values, axis=0)
        i=i+1

    B = ['row', 'col']
    B.extend(columns)
    properties = pd.DataFrame(data=properties, columns=B)
    #print(properties)
    fig = make_subplots(rows=math.ceil(math.sqrt(data.shape[1])), cols=math.ceil(math.sqrt(data.shape[1])), shared_xaxes=False, horizontal_spacing=0.1, vertical_spacing=0.05, subplot_titles=columns, column_widths=None, row_heights=None)
    i=0
    zmin=min(np.min(properties.iloc[:,2:]))
    zmax=max(np.max(properties.iloc[:,2:]))
    for property in columns:
        fig.add_traces(
            [go.Heatmap(z=properties.sort_values(by=['row', 'col'])[property].values.reshape(size,size), zmax=zmax, zmin=zmin, coloraxis = 'coloraxis2')],
            rows=[i // math.ceil(math.sqrt(data.shape[1])) + 1 ],
            cols=[i % math.ceil(math.sqrt(data.shape[1])) + 1 ]
            )
        i=i+1

    for layout in fig.layout:
        if layout.startswith('xaxis') or layout.startswith('yaxis'):
            fig.layout[layout].visible=False
            fig.layout[layout].visible=False
        if layout.startswith('coloraxis'):
            fig.layout[layout].cmax=zmax
            fig.layout[layout].cmin=zmin
        if layout.startswith('colorscale'):
            fig.layout[layout]={'diverging':'viridis'}

    fig.update_layout(
        height=800
    )
    fig.show()
    #fig.savefig('C:/Users/ack98/Downloads/hou_likeness/NeuronVals1_19.png')
    
showPropertyPlot(som, X, feature_names)


# In[23]:


#get abbreviation for census tracts

#data2=data.astype(str)
#print(data)

tractlabel = pd.Series(data.Tract.values,index=data.GEOID).to_dict()


# In[27]:


def shorten_tract(c):
    if c > 0:
        return tractlabel[c]
    else:
        return c

CT_map = som.labels_map(X, data.GEOID)

plt.figure(figsize=(14, 14))
for p, tracts in CT_map.items():
    tracts = list(tracts)
    x = p[0] + .1
    y = p[1] - .3
    for i, c in enumerate(tracts):
        off_set = (i+1)/len(tracts) - 0.05
        plt.text(x, y+off_set, shorten_tract(c),color=colors_dict[c], fontsize=10)
plt.pcolor(som.distance_map().T, cmap='gray_r', alpha=.2)
plt.xticks(np.arange(size+1))
plt.yticks(np.arange(size+1))
plt.grid()

legend_elements = [Patch(facecolor=clr,
                         edgecolor='w',
                         label=l) for l, clr in maxlabel_color.items()]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
plt.show()


# In[68]:


#now get the magnitude of weights associated for each neuron

W = som.get_weights()
#print(W)
plt.figure(figsize=(10, 10))
for i, f in enumerate(feature_names):
    plt.subplot(3, 3, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
    #print(W)
plt.tight_layout()
plt.show()


# In[87]:


#In this map we associate each neuron to the feature with the maximum weight. 
#This segments our map in regions where specific features have high values.
Z = np.zeros((size, size))
plt.figure(figsize=(8, 8))
for i in np.arange(som._weights.shape[0]):
    for j in np.arange(som._weights.shape[1]):
        #W = som.get_weights()
        #print(W)
        feature = np.argmax(W[i, j , :])
        plt.plot([j+.5], [i+.5], 'o', color='C'+str(feature),
                 marker='s', markersize=24)
        

legend_elements = [Patch(facecolor='C'+str(i),
                         edgecolor='w',
                         label=f) for i, f in enumerate(feature_names)]

plt.legend(handles=legend_elements,
           loc='center left',
           bbox_to_anchor=(1, .95))
        
plt.xlim([0, size])
plt.ylim([0, size])
plt.show()


# In[ ]:




