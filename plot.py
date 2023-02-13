# Plots the values obtained from the basline training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_clustered_stacked(dfall, labels=None, title="Policy Refinement Time",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe

def plot_percentage():
	barWidth = 0.25
	fig = plt.subplots(figsize =(12, 8))
	
	# set height of bar
	IT = [37, 28, 15]
	ECE = [83.4, 63, 33]
	CSE = [100, 100, 94.5]
	
	# Set position of bar on X axis
	br1 = np.arange(len(IT))
	br2 = [x + barWidth for x in br1]
	br3 = [x + barWidth for x in br2]
	
	# Make the plot
	plt.bar(br1, IT, color ='r', width = barWidth,
			edgecolor ='grey', label ='Policy A')
	plt.bar(br2, ECE, color ='g', width = barWidth,
			edgecolor ='grey', label ='Policy B')
	plt.bar(br3, CSE, color ='b', width = barWidth,
			edgecolor ='grey', label ='Policy C')
	
	# Adding Xticks
	plt.xlabel('Environment', fontweight ='bold', fontsize = 15)
	plt.ylabel('% Counterexamples Corrected', fontweight ='bold', fontsize = 15)
	plt.xticks([r + barWidth for r in range(len(IT))],
			['MultiWalker', 'CACC', 'MultiAnt'])
	
	plt.legend()
	plt.show()

# create fake dataframes
df1 = pd.DataFrame(np.array([[3500000,0,0],
 [1200000,0,0],
 [3000000,0,0]]),
                   index=["MultiWalker", "CACC", "ANT"],
                   columns=["Primary", "Update", "Seconday"])
df2 = pd.DataFrame(np.array([[1800000,45000,0],
 [1200000,18000,0],
 [2400000,50000,0]]),
                   index=["MultiWalker", "CACC", "ANT"],
                   columns=["Primary", "Update", "Seconday"])
df3 = pd.DataFrame(np.array([[1800000,45000,45000],
 [1000000,18000,18000],
 [2000000,50000,500000]]),
                   index=["MultiWalker", "CACC", "ANT"], 
                   columns=["Primary", "Update", "Seconday"])

# Then, just call :
if __name__ == "__main__":
	plot_clustered_stacked([df1, df2, df3],["Policy A", "Policy B", "Policy C"])
	#print(np.random.rand(3, 3))
	plt.show()
	plot_percentage()