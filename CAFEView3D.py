# CAFEView3D.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class CAFEView:

  def __init__(self, filename, sep='\t', resolution=10):
    self.filename=filename
    self.sep=sep
    self.resolution=resolution

  def buildDF(self):
    self.df=pd.read_csv(self.filename, sep=self.sep)
    self.df=self.df[(self.df['X']*1000%self.resolution==0) & (self.df['Y']*1000%self.resolution==0) & (self.df['Z']*1000%self.resolution==0)]
    print('Dataframe shape: ',self.df.shape)

  def plotDF(self, response='Material'):
    # axes instance
    self.fig = plt.figure(figsize=(6,6))
    self.ax = Axes3D(self.fig, auto_add_to_figure=False)
    self.fig.add_axes(self.ax)

    # plot
    sc = self.ax.scatter(self.df['X'], self.df['Y'], self.df['Z'], s=10, c=self.df[response], marker='o', cmap='coolwarm', alpha=0.5) # alpha=intensity, s=size points
    self.ax.set_xlabel('X')
    self.ax.set_ylabel('Y')
    self.ax.set_zlabel('Z')
    self.ax.set_title(response)
    #plt.colorbar(sc)

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(2, 1), loc=2)
    plt.show()

  def printDF(self):
    #self.buildDF()
    print(self.df)

  def dropDF(self, column_='Unnamed: 15'):
    self.df.drop(column_, axis=1, inplace=True)

  def readDF(self):
    return self.df

path='~/cafe_gnn_s/MINI/ALE3DCAFE_PANDAS_iso_40/iso_40.351.dat'  
test=CAFEView(filename=path, sep='\t', resolution=10)
test.buildDF()
test.dropDF()
data=test.readDF()
test.plotDF(response='CellState')
print(data)
del test
