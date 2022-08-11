import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split


RANDOM_STATE = 1

# modified from source: https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def cart2spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def utc_str_to_timestamp(time_str):
    return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f').timestamp()

def datetime_to_seconds(datetime_value):
    return datetime_value.total_seconds()


''''
The following two functions, heatmap and annotate_heatmap, are taken from the matplot lib documentation examples (https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html)
'''

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def import_raw_data():
    train_labels = pd.read_csv('https://courses.edx.org/assets/courseware/v1/d64e74647423e525bbeb13f2884e9cfa/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/train_labels.csv',index_col=0)
    train_time_series = pd.read_csv('https://courses.edx.org/assets/courseware/v1/b98039c3648763aae4f153a6ed32f38b/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/train_time_series.csv',index_col=0)
    test_labels = pd.read_csv('https://courses.edx.org/assets/courseware/v1/72d5933c310cf5eac3fa3f28b26d9c39/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/test_labels.csv',index_col=0)
    test_time_series = pd.read_csv('https://courses.edx.org/assets/courseware/v1/1ca4f3d4976f07b8c4ecf99cf8f7bdbc/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/test_time_series.csv',index_col=0)
    
    # remove first couple entries because doesn't line up with provided labels
    train_time_series = train_time_series.iloc[4:,:]
    train_labels = train_labels.iloc[1:, :]
    
    # change UTC time to absoluate value in seconds 
    train_time_series['UTC time'] = train_time_series['UTC time'].apply(utc_str_to_timestamp)
    test_time_series['UTC time'] = test_time_series['UTC time'].apply(utc_str_to_timestamp)
    
    return train_time_series, train_labels, test_time_series, test_labels


def import_accelerometer_data(): 
    train_labels = pd.read_csv('https://courses.edx.org/assets/courseware/v1/d64e74647423e525bbeb13f2884e9cfa/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/train_labels.csv',index_col=0)
    train_time_series = pd.read_csv('https://courses.edx.org/assets/courseware/v1/b98039c3648763aae4f153a6ed32f38b/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/train_time_series.csv',index_col=0)
    test_labels = pd.read_csv('https://courses.edx.org/assets/courseware/v1/72d5933c310cf5eac3fa3f28b26d9c39/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/test_labels.csv',index_col=0)
    test_time_series = pd.read_csv('https://courses.edx.org/assets/courseware/v1/1ca4f3d4976f07b8c4ecf99cf8f7bdbc/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/test_time_series.csv',index_col=0)
    
    # remove first couple entries because doesn't line up with provided labels
    train_time_series = train_time_series.iloc[4:,:]
    train_labels = train_labels.iloc[1:, :]
    
    
    # remove columns without utility
    train_time_series.pop('accuracy')
    test_time_series.pop('accuracy')
    train_time_series.pop('timestamp')
    test_time_series.pop('timestamp')
    train_time_series.pop('UTC time')
    test_time_series.pop('UTC time')
    
    return train_time_series, train_labels, test_time_series, test_labels


def import_and_split_data(window):
    '''
    This function downloads the project dataset from a set of EDX urls. It then proceeds to normalise the data and reshape it into observations of windows of time. That means, for each label we can also use the acceleremeter readings from moments before and after the time of the reading. The size of the window will be optimized. 
    
    Parameters
    ----------
    window
        Is an integer which must be odd because the window must be a symmetric length on either side of the current time step.
        
    Returns
    -------
    train_X 
        A numpy data matrix for training a ML model.
    
    val_X 
        A numpy data matrix for ML model validation.
    
    test_X 
        The test data matrix provided in the project returned as a numpy array and processed in the same way as the training and validation data matrices. 
    
    train_Y
        A numpy array of labels from the training data set.
    
    val_Y
        A numpy array of the validation labels from the training set.
    '''
    
    
    # first check that the input is valid
    if type(window) != int:
        raise TypeError('The input window is not an integer.')
    if window%2 != 1:
        raise ValueError('The input window must be an odd number.')
    if window > 19:
        raise ValueError('The window is too large.')
    
    # import data into groups of 3 and give labels according to the groups of 10
    train_labels = pd.read_csv('https://courses.edx.org/assets/courseware/v1/d64e74647423e525bbeb13f2884e9cfa/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/train_labels.csv',index_col=0)
    train_time_series = pd.read_csv('https://courses.edx.org/assets/courseware/v1/b98039c3648763aae4f153a6ed32f38b/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/train_time_series.csv',index_col=0)
    test_labels = pd.read_csv('https://courses.edx.org/assets/courseware/v1/72d5933c310cf5eac3fa3f28b26d9c39/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/test_labels.csv',index_col=0)
    test_time_series = pd.read_csv('https://courses.edx.org/assets/courseware/v1/1ca4f3d4976f07b8c4ecf99cf8f7bdbc/asset-v1:HarvardX+PH526x+2T2020+type@asset+block/test_time_series.csv',index_col=0)

    # remove first three entires in training set (oddity of the particular dataset, the remaining readings come in groups of 10 with one reading)
    train_labels = train_labels.iloc[1:, :]
    train_time_series = train_time_series.iloc[4:,:]

    # pop off unecassary features
    train_time_series.pop('accuracy')
    test_time_series.pop('accuracy')

    train_time_series.pop('UTC time')
    test_time_series.pop('UTC time')

    train_time_series.pop('timestamp')
    test_time_series.pop('timestamp')

    # normalise accelerometer readings
    train_time_series[['x','y','z']] = (train_time_series[['x','y','z']] - train_time_series[['x','y','z']].mean())/(train_time_series[['x','y','z']].max() - train_time_series[['x','y','z']].min())
    test_time_series[['x','y','z']] = (test_time_series[['x','y','z']] - test_time_series[['x','y','z']].mean())/(test_time_series[['x','y','z']].max() - test_time_series[['x','y','z']].min())

    # organise into numpy array for training
    train_X = train_time_series.to_numpy()
    test_X = test_time_series.to_numpy()

    # assume change of activity cannot be instananeous and copy over label in groups of 10
    train_Y = np.zeros(train_X.shape[0])
    test_Y = np.zeros(test_X.shape[0])
    
    for i in range(len(train_labels['label'])):
        first_index = i*10
        second_index = first_index+10
        train_Y[first_index:second_index] = np.ones(10) * (train_labels['label'].iloc[i])
    
    if window == 1:
        pass
    else:
        starting_index = int(np.floor(window/2))
        
        # copy over left hand side of window first
        tmp_train =  train_X[0:1-window,:]
        tmp_test =  test_X[0:1-window,:]
        
                              
        # copy over the remaining values in the window
        for i in range(1,window-1):
            tmp_train = np.concatenate((tmp_train, train_X[i:i+1-window]),axis=1) # copy over horizontally 
            tmp_test = np.concatenate((tmp_test, test_X[i:i+1-window]),axis=1) # copy over horizontally 
        
        # copy over last observation in the window
        train_X = np.concatenate((tmp_train, train_X[window-1:]),axis=1) 
        test_X = np.concatenate((tmp_test, test_X[window-1:]),axis=1) 
        
        # trim labels to match up with training observations
        train_Y = train_Y[starting_index:-starting_index]
    
    # split train_X into training and validation sets
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y,random_state=RANDOM_STATE)
        # use random_state=1 to have the same training/validation split across different runs
    
    return train_X, val_X, test_X, train_Y, val_Y