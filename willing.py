import pandas as pd
def ta_df(filename):
    '''Accepts a TA preferences file and returns it as a dataframe'''
    try:
        return (pd.read_csv(filename, 
                            header = None))
    except:
        print('invalid file')

def _pref_df(filename):
    df = ta_df(filename
               ).iloc[1:,3:]
    df = df.reset_index(drop = True).set_axis(range(df.shape[1]), axis=1)
    return df.replace(to_replace=['W','U','P'], value =[1,0,0])


def willing(test_filename, ta_filename):
    ta_df = _pref_df(ta_filename)
    test_df = pd.read_csv(test_filename, header=None)
    # Count matching 1s across all elements
    return ((test_df == 1) & (ta_df == 1)).sum().sum()
