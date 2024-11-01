import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.model_selection import *
from sklearn.metrics import *
from scipy import integrate
from scipy.fft import rfft
from datetime import datetime
from scipy.stats import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, Ridge

'''
    load_features(vals)

    - n: train file number (window in the time series) [integer - range{0-1099}]
    - test: returns test values instead of train [boolean - default=False]
    - vals: load as pd Dataframe or np 2d-array [boolean - default=False (df)]
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

    - vals: load as pd Dataframe or np 2d-array [boolean - default=False (df)]
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

    - vals: load as pd Dataframe or np 2d-array [boolean - default=False (df)]
'''
def load_wave_features(vals = False, test = False):
    
    if test:
        filename = 'wave_features_test'
    else:
        filename = 'wave_features'
        
    filename = os.path.join('../data', filename + '.csv')
    features = pd.read_csv(filename)
    
    if vals:
        features = features.values[:, 1:]

    return features.iloc[: , 1:]

''' 
    plot_sorted_counts(data, label, xtick, rot, sorted, coloring)

    - data: Data to plot [pd Dataframe]
    - label: Y-axis label (variable to count) [string] 
    - xtick: Show Xticks [boolean - default=True]
    - rot: Rotation of Xticks [double - default=90]
    - sorted: Sort in descending order [boolean - default=True]
    - coloring: Coloring of bars (based on direction, speed, or default coloring) [string - possible values: 'spd', 'dir', None - default=None]
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
        print(values,counts)

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

    - start: wave window start index [integer]
    - end: wave window end index [integer]
    - low: plot low wave [boolean - default=False]
    - high: plot high wave [boolean - default=False]
    - acc: plot acceleration wave [boolean - default=True]
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
    get_wave_stats(wave):

    - wave: integer specifying which wave (low, high, acceleration, speed, position, rfft_low, rfft_high, rfft_acceleration) [integer - {0, 1, 2, 3, 4, 5, 6, 7}]
'''
def get_wave_stats(wave, test=False):

    column_names = ['min_amplitude','max_amplitude', 'mean_aplitude', 'variance', 'q_25', 'median_amplitude', 'q_75', 'kurtosis', 'skewness', 'rms_amplitude', 'crest_factor']

    # Imputer for filling missing (NaN) values with regression
    imputer = IterativeImputer(estimator=Ridge(), random_state=2211595)
    
    # EMpty dfs for data accumulation
    compressed_df = pd.DataFrame()
    hist_df = pd.DataFrame()
    
    for i in range(1100):
        
        if test:
            X = load_features(i, test=True)
            
        else:
            X = load_features(i)
                
        if wave == 3:
            # If wave index is 3, compute speed
            speed = np.real(integrate.cumulative_trapezoid(X[str(wave-1)], x = X.iloc[:,0]))
            X['3'] = pd.DataFrame(np.transpose(speed))
        
        elif wave == 4:
            # If wave index is 4, compute speed then position
            speed = np.real(integrate.cumulative_trapezoid(X[str(wave-2)], x = X.iloc[:,0]))            
            position = np.real(integrate.cumulative_trapezoid(speed, x = X.iloc[1:,0]))
            
            X['4'] = pd.DataFrame(np.transpose(position))        
        
        elif wave >= 5:
            # If wave index is greater than or equal to 5, compute the respective rfft of a wave (low, high, acc)
            rft = np.abs(rfft(X[str(wave-5)].values-np.mean(X[str(wave-5)].values)))
            X[str(wave)] = pd.DataFrame(np.transpose(rft))

        # Drop first useless column (Unnamed)
        X = X.drop(X.columns[0], axis=1)

        # Counter to keep track of progress
        if i % 200 == 0:
            p = round(((wave * 1100) + i ) / 8800 * 100)
            print(f'Wave {wave} at index {i} - Progress: {p}%' )

        # Impute missing values by mean
        X.iloc[:, min(wave,3)] = imputer.fit_transform(X.iloc[:, min(wave,3)].values.reshape(-1,1))
        
        # Extract the relevant wave 
        sine_wave = X.iloc[:, min(wave,3)]

        # Extract Compressed features from each wave
        kurt = sine_wave.kurtosis()
        skewness = sine_wave.skew()
        min_amplitude = np.min(sine_wave)
        max_amplitude = np.max(sine_wave)
        mean_amplitude = np.mean(sine_wave)
        variance = np.var(sine_wave)
        q_25 = sine_wave.quantile(q=0.25)
        median_amplitude = sine_wave.quantile(q=0.5)
        q_75 = sine_wave.quantile(q=0.75)
        rms_amplitude = np.sqrt(np.mean(sine_wave**2))
        crest_factor = max_amplitude / rms_amplitude

        # Create list of features to then create a df with the according columns 
        compressed = [min_amplitude, max_amplitude, mean_amplitude, variance, q_25, median_amplitude, q_75, kurt, skewness,  rms_amplitude, crest_factor]
        compressed_features = pd.DataFrame([compressed], columns=[f'{col}' for col in column_names])

        compressed_df = pd.concat([compressed_df, compressed_features], axis = 0)
        
        # Create the histogram of the wave, then create a df with the according columns
        hist, bin_edges = np.histogram(sine_wave, bins=1100)
        hist_features = pd.DataFrame([hist], columns=[f'h{col}' for col in range(len(hist))])
            
        # hist_features.columns = [waves[wave] + '_' + col for col in hist_features.columns]
        hist_df = pd.concat([hist_df, hist_features], axis = 0)

    # Concatenate the 2 dataframes into a final feature set for a specific wave     
    wave_df = pd.concat([compressed_df, hist_df], axis=1)
    
    return wave_df

'''
    store_wave_features(store)

    - store: Stores the obtained wave features in a excel file [boolean - default=False]
'''
def store_wave_features(store = False, test=False):
    
    waves = ['low', 'high', 'acc', 'spd', 'pos', 'rft0', 'rft1', 'rft2']
    df = pd.DataFrame()
    
    for i in range(len(waves)):
        wave_stats = get_wave_stats(i, test=test)
            
        wave_stats.columns = [waves[i] + '_' + col for col in wave_stats.columns]
        df = pd.concat([df, wave_stats], axis = 1)

    if store and test:
        file_path = '..//data//wave_features_test.csv'
        df.to_csv(file_path, index=True, mode='w')
        
    elif store:
        file_path = '..//data//wave_features.csv'
        df.to_csv(file_path, index=True, mode='w')
    
    print("\n-----Done-----")
    
    return df


'''
    get_correlations(X) 

    - X: Dataframe to display feature correlation in descending order [pd Dataframe]
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

    - X: Complete wave features dataset to analyze [pd Dataframe]
    - features: Subset of features for analysis [list of strings]
    - title: Name of preprocessing technique if used, or just Original data [string]
    - labels: Labels assigned to each window [np array]
    - scaler: Visualize the scaling behaviour [sklearn Scaling technnique - default=None (original data)]
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
    
    
'''
    Function used in lab. Used in error analysis function
'''
def plot_learning_curve(sizes, train, val):

    train_scores_mean = np.mean(train, axis=1)
    train_scores_std = np.std(train, axis=1)
    val_scores_mean = np.mean(val, axis=1)
    val_scores_std = np.std(val, axis=1)

    _, axes = plt.subplots(1,)
    axes.grid()
    
    axes.fill_between(
        sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="g",
    )
    
    axes.fill_between(
        sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color="r",
    )
    
    axes.plot(
        sizes, train_scores_mean, "o-", color="g", label="Training score"
    )
    
    axes.plot(
        sizes, val_scores_mean, "o-", color="r", label="Cross-validation score"
    )
    
    axes.set_ylim(-0.26, -0.17)
    
    axes.legend(loc="best")
    
    plt.title('Learning Curve')
    plt.show()
    
    return


'''
    error_analysis(model, Xtrain, Ytrain, cv)

    - model: model pipeline used for error analysis [sklearn model pipeline]
    - Xtrain: Training values [pd Dataframe]
    - Ytrain: Training labels [pd Dataframe]
    - cv: Cross validation technique used [sklearn technique]
'''
            
def error_analysis(model, Xtrain, Ytrain, cv):
    
    '''
    Plotting learning curve
    '''
    # Get learning curve values
    train_sizes, train_scores, val_scores = learning_curve(model, Xtrain, Ytrain, cv=cv, n_jobs=4, verbose=0)
    
    # Plot the curve
    plot_learning_curve(train_sizes, train_scores, val_scores)

    '''
    Error analysis
    '''
    # Get validation results from cross validation
    Yval = cross_val_predict(model, Xtrain, y=Ytrain, cv=cv, n_jobs=4, verbose=0)
    
    # Get error analysis values from validation results
    p, r, F, sup = precision_recall_fscore_support(Ytrain, Yval)
    validation_results = pd.DataFrame({'precision':p,'recall':r,'F1-score':F,'support':sup})

    # Add the group labels for each of the errors analysis values
    groups = pd.DataFrame({'group': model.classes_})
    
    validation_results = pd.concat([validation_results, groups], axis = 1)
    validation_results.set_index('group', inplace=True)
    validation_results.sort_values(by='precision', inplace=True, ascending=False)
    
    print('\n\n')


    ''' 
    Plot bars for error analysis 
    '''
    
    ax = validation_results[['precision', 'recall', 'F1-score']].plot.bar(figsize=(10, 6), rot=0)

    for i, (index, row) in enumerate(validation_results.iterrows()):
        for j, value in enumerate(row[['precision', 'recall', 'F1-score']]):
            if j == 0:
                ax.text(i + j * 0.2, value - 0.1, f"{int(row['support'])}", ha='center', va='bottom', fontsize=20, color='k')

    ax.set_xlabel('Group')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1-score, and Support by Group label')

    handles, labels = ax.get_legend_handles_labels()
    support_legend_entry = Line2D([0], [0], marker='', color='k', label='Support', markersize=10, linestyle='-')

    handles.append(support_legend_entry)
    labels.append('Support')

    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
    
    
    print('\n\n')

    
    '''
    Plot scatter for error analysis
    '''
    
    validation_results.plot.scatter(x='precision', y='recall', c='support', colormap='viridis', s=50)

    plt.tight_layout()
    plt.title('Precision and Recall scatter')
    plt.show()
    
    
    print('\n\n')
    
    
    
    '''
    Plot Confusion Matrix
    '''

    fig, ax = plt.subplots(figsize=(11, 8))

    ConfusionMatrixDisplay.from_predictions(Ytrain, Yval, include_values=True,normalize='true', xticks_rotation=45, ax=ax)

    plt.title('Confusion Matrix')
    plt.show()
    
    
     
"""
    regression_error_analysis(model, Xtrain, Ytrain, cv)
    
    - model: Model pipeline used for error analysis [sklearn model pipeline]
    - Xtrain: Training values [pd Dataframe]
    - Ytrain: Training labels [pd Dataframe]
    - cv: Cross validation technique used [sklearn technique]
"""

def regression_error_analysis(model, Xtrain, Ytrain, cv):

    '''
    Plotting learning curve
    '''
    # Get learning curve values
    train_sizes, train_scores, val_scores = learning_curve(model, Xtrain, Ytrain, cv=cv, n_jobs=4, verbose=0)
    
    # Plot the curve
    plot_learning_curve(train_sizes, train_scores, val_scores)


    
    '''
    Residuals analysis
    '''
    # Get validation results from cross validation
    YpredCV = cross_val_predict(model, Xtrain, Ytrain, cv=cv)

    residuals = Ytrain - YpredCV

    plt.figure(figsize=(9, 8))

    plt.scatter(YpredCV, residuals, color='blue', marker='o', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')



    # Extract the best regressor inside the model pipeline
    regressor = model.best_estimator_.named_steps['regressor'][0]


    
    ''' 
    Plot bar histogram of feature importance
    '''

    # Extract feature importance coefficients from the regression model
    feature_importance = np.abs(regressor.coef_)
    feature_names = np.array(Xtrain.columns)
    
    df = pd.DataFrame({'feature_names':feature_names,'feature_importance':feature_importance})
    df.sort_values(by=['feature_importance'], ascending=True, inplace=True)
    
    plt.figure(figsize=(12, 18))
    plt.barh(df['feature_names'], df['feature_importance'], color='green', alpha=0.7)
    plt.title('Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')

    plt.show()


    
    '''
    Cross validation interval analysis
    '''
    residual_standard_error = np.std(residuals)

    # Calculate the t-statistic for a desired confidence level (e.g., 95%)
    confidence_level = 0.95
    degrees_of_freedom = len(Xtrain) - 2  # Adjust for intercept and one predictor
    t_statistic = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    
    # Calculate the margin of error
    margin_of_error = t_statistic * residual_standard_error
    
    # Construct the prediction interval
    lower_bound = YpredCV - margin_of_error
    upper_bound = YpredCV + margin_of_error

    # Sort the indices of Ytrain, then adapt YpredCV and the respective margins of predictions
    sorted_indices = np.lexsort((YpredCV, Ytrain))

    Ytrain = Ytrain[sorted_indices]
    YpredCV = YpredCV[sorted_indices]
    
    lower_bound = lower_bound[sorted_indices]
    upper_bound = upper_bound[sorted_indices]

    
    # Visualize the prediction intervals
    plt.figure(figsize=(30, 18))
    plt.scatter(range(len(Ytrain)), Ytrain, label='Actual', color='blue', marker='o')
    plt.scatter(range(len(YpredCV)), YpredCV, label='Predicted', color='orange', linestyle='-', marker = 'o')
    plt.errorbar(range(len(YpredCV)), YpredCV, yerr=margin_of_error, fmt='o', color='orange', alpha=0.2, label='Prediction Interval')


    # Text labels for Ytrain labels
    Ylabels = np.linspace(-0.5, 0.5, 11)
    
    for i, Ypos in enumerate(Ylabels):
        Xpos = len(Ytrain) // 11 * i + 30
        plt.text(Xpos, -0.75 + i * 0.06, f'{Ypos:.1f}', fontsize=30, ha='center', va='center',color='b',weight = 'black')

    plt.title('Prediction Intervals with Cross-Validation', fontsize=24)
    
    plt.xlabel('Data Point', fontsize=20)
    plt.xticks(fontsize=17)
    
    plt.ylabel('Target Variablw', fontsize=20)
    plt.yticks(fontsize=17)
    
    plt.legend(fontsize=17)
    
    plt.tight_layout()
    plt.show()


    
    # '''
    # Partial Dependence Plot
    # '''
    
    # # Create partial dependence plot
    # PartialDependenceDisplay.from_estimator(regressor, Xtrain, df['feature_names'].tail(1), grid_resolution=50)
    # plt.suptitle(f'Partial Dependence Plot for Feature {feature_index}')
    # plt.subplots_adjust(top=0.9)  # Adjust title position
    # plt.show()


    
    ''' 
    Display other useful error metrics
    '''
    mae_cv = mean_absolute_error(Ytrain, YpredCV)
    mse_cv = mean_squared_error(Ytrain, YpredCV)
    r2_cv = r2_score(Ytrain, YpredCV)

    print(f'Cross-Validation Mean Absolute Error: {mae_cv:.4f}')
    print(f'Cross-Validation Mean Squared Error: {mse_cv:.4f}')
    print(f'Cross-Validation R-squared: {r2_cv:.4f}')
    
'''
predict_and_store(model, feature_extraction)
model: Best model pipeline after training, used for predicting [sklearn model pipeline]
feature_extraction: Function used to extract features from training data [python function]
''' 
## WARNING: The 'feature_extraction' parameter currently only works on my own function
##          because of the specified parameters of the function I used
def predict_and_store(model, feature_extraction):
    
    # Loads test data and extract the features, similarl to the train data
    Xtest = feature_extraction(store = False, test = True)    
    
    # Use the model to make the predictions based on the test data
    Ytest = model.predict(Xtest)
    
    # Create the dataframe and prepare it for submission
    predictions = pd.DataFrame({'misalignment': Ytest})
    predictions['id'] = range(len(predictions))
    predictions = predictions[['id', 'misalignment']]
    predictions['misalignment'] = predictions['misalignment'].astype(float)

    # Name the prediction file with timestamp 
    output_directory = '..//submissions'
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"output_{timestamp}.csv"
    
    # Store the file in the submission folder
    predictions.to_csv(os.path.join(output_directory, output_filename), index=False)
