from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from graphviz import Source
from IPython.display import SVG,display
from sklearn.tree import export_graphviz



def plot_tree(tree_classifier,data_X,data_y,max_depth=None,proportion=False, rotate=False):
    dtree_graph = Source(export_graphviz(tree_classifier,
                                             out_file=None, 
                                             feature_names = [str(x) for x in list(data_X.columns.values)],
                                             class_names = [str(x) for x in list(data_y.unique())],
                                             rounded = True,
                                             proportion = proportion, 
                                             precision = 2,
                                             filled = True,
                                             max_depth=max_depth,
                                             rotate=rotate))

    return dtree_graph.pipe(format='png')
    return display(SVG(dtree_graph.pipe(format='svg')))


    





def plot_histograms(df_org,graph_per_row=3,max_unique=50):
    unique_n =  df_org.nunique()
    df_num = df_org._get_numeric_data()
    # filter out ids, non-numerical with more than max_unique values
    column_names_to_plot = df_org[[col for col in df_org if (unique_n[col] > 1 and unique_n[col] < max_unique) or (col in df_num.columns and unique_n[col]<len(df_org))]].columns 
    # calculate number of rows
    graph_rows = int(np.ceil(len(column_names_to_plot)/graph_per_row))
    # create subplots
    fig, axes = plt.subplots(graph_rows, graph_per_row, figsize=(6*graph_per_row, 6*graph_rows))
    # flatten
    axes = [val for sublist in axes for val in sublist]
    # plot each column separately
    for column_name,axis in zip(column_names_to_plot,axes):
        # numerical columns
        if column_name in df_num.columns:
            sns.distplot(df_org[column_name].dropna(),ax=axis,hist=True)
        # non-numerical
        else:
            df_dummy = df_org[column_name].value_counts()/len(df_org[column_name])
            df_dummy.plot.bar(ax=axis)
        axis.set_title(column_name)


def df_one_hot(df_org_train,df_org_test,categorical_columns=[]):

    # make a copy of the dataframes
    df_train_cp = df_org_train.copy()
    df_test_cp = df_org_test.copy()

    # remember for later (potentially removing additional colummns
    train_cols =  set(df_train_cp.columns.values)
    test_cols  =  set(df_test_cp.columns.values)
    
    df_train_cp['split'] = 'train'
    df_test_cp['split']  = 'test'
    
    df_all_cp    = df_train_cp.append(df_test_cp,sort=False)
    df_all_cp    = pd.get_dummies(df_all_cp,prefix_sep='__',columns=categorical_columns)

    df_new_train = df_all_cp[df_all_cp['split']=='train']
    df_new_test  = df_all_cp[df_all_cp['split']=='test']

    # remove additional columns
    df_new_train = df_new_train.drop(['split']+list(test_cols-train_cols),axis=1)
    df_new_test  = df_new_test.drop(['split']+list(train_cols-test_cols),axis=1)

    return df_new_train,df_new_test


    


        
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df_org, nGraphShown, nGraphPerRow):
    nunique = df_org.nunique()
    df = df_org[[col for col in df_org if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df_org.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df_org, graphWidth):
    filename = df.dataframeName
    df = df_org.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

    

