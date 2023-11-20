import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew
from matplotlib.colors import LinearSegmentedColormap

'''
load_features(vals)

n: train file number (window in the time series) [integer - range{0-1099}]
test: returns test values instead of train [boolean - default=False]
vals: load as pd Dataframe or np 2d-array [boolean - default=False (df)]
'''
def load_features(n, test = False, vals = False):
    
    if test:
        filename = 'test'+ '_' + str(n)
        filename = os.path.join('../data', 'test', filename + '.csv')
        
    else:
        filename = 'train' + '_' + str(n)
        filename = os.path.join('../data', 'train', filename + '.csv')

        
    features = pd.read_csv(filename)
    
    if vals:
        features = features.values[:, 1:]

    return features

'''
load_train_meta(vals)

vals: load as pd Dataframe or np 2d-array [boolean - default=False (df)]
'''
def load_train_meta(vals = False):
    filename = 'train_meta'
    filename = os.path.join('../data', filename + '.csv')
    
    features = pd.read_csv(filename)

    if vals:
        features = features.values[:, 1:]
        
    return features

'''
load_wave_features(vals)

vals: load as pd Dataframe or np 2d-array [boolean - default=False (df)]
'''
def load_wave_features(vals = False):
    filename = 'wave_features'
    filename = os.path.join('../data', filename + '.csv')
    
    features = pd.read_csv(filename)
    
    if vals:
        features = features.values[:, 1:]

    return features.iloc[: , 1:]

''' 
plot_sorted_counts(data, label, xtick, rot, sorted, coloring)

data: Data to plot [pd Dataframe]
label: Y-axis label (variable to count) [string] 
xtick: Show Xticks [boolean - default=True]
rot: Rotation of Xticks [double - default=90]
sorted: Sort in descending order [boolean - default=True]
coloring: Coloring of bars (based on direction, speed, or default coloring) [string - possible values: 'spd', 'dir', None - default=None]
'''
def plot_sorted_counts(data, label, xtick=True, rot=90, sorted=True, coloring=None):

    # Count values then sort them in descending order
    if sorted:
        values, counts = np.unique(data, return_counts=True)
        
        sorted_indices = np.argsort(-counts)
        
        values = values[sorted_indices].astype(str)
        counts = counts[sorted_indices]
        
    # Count values without sorting
    else:
        values, counts = np.unique(data, return_counts=True)

    # Adjust size
    plt.figure(figsize=(10, 6))
    
    ax = plt.axes()
    ax.set_facecolor("g")

    # Adjust title
    title = ''
    
    if sorted:
        title += " counts in descending order"
    else:
        title += " counts"

    # Adjust coloring based on parameter (depending on what we want to show)
    color_cycle = plt.cm.flag
    
    if coloring == 'dir':
        bar_color = [color_cycle(0) if val.startswith('H') else color_cycle(0.3) for val in values]
        title += " colored based on direction"
        
    elif coloring == 'spd':
        bar_color = [color_cycle(0) if val.endswith('425') else color_cycle(0.3) for val in values]
        title += " colored based on speed"

    else:
        bar_color = color_cycle(np.linspace(0, 1, len(values)))

    # Plot bars
    plt.bar(values, counts, color=bar_color)   

    # Plot title
    plt.title(label + title)

    # Adjust Xticks
    if xtick:
        plt.xlabel(label)
        plt.xticks(rotation=rot, fontsize=7.5)
    else:
        plt.xticks([], [])

    plt.ylabel('Counts')

    plt.show()
    
    
'''
plot_series(start, end, low, high, acc)

start: wave window start index [integer]
end: wave window end index [integer]
low: plot low wave [boolean - default=False]
high: plot high wave [boolean - default=False]
acc: plot acceleration wave [boolean - default=True]
'''
def plot_series(start, end, low = False, high = False, acc = True):
    
    plt.figure(figsize=(12, 8))
    
    for i in range(start, end):
        df = pd.DataFrame(load_features(i, True))

        lo = df['0']
        hi = df['1']
        ac = df['2']

        if low:
            plt.plot(df.index, lo, label='Clamp Low Resolution '+str(i))

        if high:
            plt.plot(df.index, hi, label='Clamp High Resolution '+str(i))
            
        if acc:
            plt.plot(df.index, ac, label='Acceleration Signal '+str(i))

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Plot')
    plt.legend()
    plt.show()
    
'''
get_wave_stats(window_start, window_end, wave):

window_start: wave window start index [integer]
window_end: wave window end index [integer]
wave: integer specifying which wave (low, high, acceleration) [integer - {0, 1, 2}]
'''
def get_wave_stats(window_start, window_end, wave):
    
    column_names = ['min_amplitude','max_amplitude', 'mean_aplitude', 'q_25', 'median_amplitude', 'q_75', 'kurtosis', 'skewness', 'rms_amplitude', 'crest_factor']

    data_accumulator = []

    for i in range(window_start, window_end):
        
        X = load_features(i)
        
        X = X.drop(X.columns[0], axis=1)

        if i % 150 == 0:
            print('Window', i)
        
        sine_wave = X.iloc[:, wave]
            
        kurt = kurtosis(sine_wave)
        
        skewness = skew(sine_wave)

        min_amplitude = np.min(sine_wave)
        max_amplitude = np.max(sine_wave)

        mean_amplitude = np.mean(sine_wave)
        
        q_25 = np.percentile(sine_wave, 25)
        median_amplitude = np.median(sine_wave)
        q_75 = np.percentile(sine_wave, 75)

        rms_amplitude = np.sqrt(np.mean(sine_wave**2))
            
        crest_factor = max_amplitude / rms_amplitude
    
        row = [min_amplitude, max_amplitude, mean_amplitude, q_25, median_amplitude, q_75, kurt, skewness,  rms_amplitude, crest_factor]
        
        data_accumulator.append(row)

    
    df = pd.DataFrame(data_accumulator, columns=column_names)
    
    return df

'''
store_wave_features(store)

store: Stores the obtained wave features in a excel file (works if file does not exist) [boolean - default=False]
'''
def store_wave_features(store = False):
    
    waves = ['low', 'high', 'acc']
    df = pd.DataFrame()
    
    for i in range(3):
        
        wave_stats = get_wave_stats(0, 1100, i)
        wave_stats.columns = [waves[i] + '_' + col for col in wave_stats.columns]
        df = pd.concat([df, wave_stats], axis = 1)

    
    if store:
        file_path = 'data//wave_features.csv'
        df.to_csv(file_path, index=True, mode='w')
    
    return df


'''
get_correlations(X) 

X: Dataframe to display feature correlation in descending order [pd Dataframe]
'''
def get_correlations(X):

    cols = X.columns

    correlation_matrix = X.corr()
    correlation_list = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            correlation_value = correlation_matrix.loc[col1, col2]
            correlation_list.append((col1, col2, correlation_value))
    
    correlation_list.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for col1, col2, correlation_value in correlation_list:
        print(f"Columns: {col1} and {col2} - Correlation: {correlation_value}")
        
        
'''
analyze(X, features, title, labels, scaler)

X: Complete wave features dataset to analyze [pd Dataframe]
features: Subset of features for analysis [list of strings]
title: Name of preprocessing technique if used, or just Original data [string]
labels: Labels assigned to each window [np array]
scaler: Visualize the scaling behaviour [sklearn Scaling technnique - default=None (original data)]
'''
def analyze(X, features, title, labels, scaler=None):

    # Apply scaling technique for visualization if applicable 
    if scaler is not None:
        d_scaled = scaler.fit_transform(X)
        d = pd.DataFrame(data=d_scaled, columns=X.columns)
        
    # If not, just copy original data
    else:
        d = X.copy()

    # Relevant variables for plotting
    n = len(features)
    ids = np.arange(1100)

    # Adjust size if needed, Width x Height
    plt.figure(figsize=(12, 50))

    # Set custom colormap
    base_cmap = plt.get_cmap('bwr')
    
    lower_color = base_cmap(1.0)
    upper_color = base_cmap(0.0)

    cmap_segments = LinearSegmentedColormap.from_list('custom_cmap', [lower_color, base_cmap(0.5), upper_color], N=256)


    # Loop over the features for plotting
    for i in range(n):
        
        # Create subplots with black backgrounds
        ax = plt.subplot(n, 1, i + 1)
        ax.set_facecolor('black')

        # Scatter plot with the modified colormap
        scatter = ax.scatter(ids, d[features[i]], marker='o', c=labels, cmap=cmap_segments, vmin=-0.5, vmax=0.5)

        # Add colorbar
        cbar = plt.colorbar(scatter, orientation='vertical')
        cbar.set_label('Labels')

        plt.title(title + ' - ' + features[i])

    plt.tight_layout()
    plt.show()