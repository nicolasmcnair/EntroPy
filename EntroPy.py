from __future__ import division
import numpy as np
from math import ceil, log, exp
from itertools import product
from random import random
from scipy.signal import butter, filtfilt

def _surprise(probability):
    """Returns surprise value for given probability"""
    return log(1/probability,2)


def _probabilities(seq):
    """
    Returns a dict with each unique value in the sequence as keys and their corresponding probabilites as values.

    Args:
        seq (list): sequence.

    Returns:
        :dict:{x:probability}:
            x: unique value from sequence
            probability (float): probability of value
    """
    # Get unique values in sequence and their counts
    #unique_values,counts_values = np.unique(np.array(seq), axis=0, return_counts=True) # Need to wait until NumPy 1.13 to be able to do this
    # Return dict of x keys and their probability of occurence in the sequence
    #return dict(zip(unique_values,counts_values/np.sum(counts_values)))
    return dict([(x,seq.count(x)/len(seq)) for x in set(seq)])

def _joint_probabilities(seq, y_index):
    """
    Calculate joint probabilities of x and y in a sequence.

    Args:
        seq (list): sequence.
        y_index (int or list): relative index of y from x.

    Returns:
        :dict:{(x,y):joint_probability}:
            x: x value
            y: y value
            joint_probability (float): joint probability of values
    """
    # If y_index is int, convert to list so it can be handled using comprehension
    if isinstance(y_index, int):
        y_index = [y_index]
    # Max of y indices will determine how much the sequence needs to be truncated for x and other y indices
    offset = max(y_index,key=lambda x: abs(x))
    # Get x and y values from sequence
    x_values = seq[-offset:] if offset < 0 else seq[:-offset]
    y_values = [seq[(-offset + y):y] if y < 0 else seq[y:(len(seq) - (1 - (offset + y)))] for y in sorted(y_index,key=lambda x: abs(x))]
    # Get x,y keys
    xys = tuple(product(np.unique(x_values),tuple(product(*[set(y) for y in y_values]))))
    # Get x,y sequences
    xy_seq = zip(x_values,*y_values)
    # Return dict of x,y keys and their probability of occurence in the sequence
    return dict([((x,y),xy_seq.count((x,) + y)/len(xy_seq)) for x,y in xys])

def _conditional_probabilities(seq, y_index):
    """
    Calculate conditional probabilities of x and y in a sequence.

    Args:
        seq (list): sequence.
        y_index (int or list): relative index of y from x.

    Returns:
        :dict:{(x,y):conditional_probability}:
            x: x value
            y: y value
            conditional_probability (float): conditional probability of values
    """
    # If y_index is int, convert to list so it can be handled using comprehension
    if isinstance(y_index, int):
        y_index = [y_index]
    # Max of y indices will determine how much the sequence needs to be truncated for x and other y indices
    offset = max(y_index,key=lambda x: abs(x))
    # Get x and y values from sequence
    x_values = seq[-offset:] if offset < 0 else seq[:-offset]
    y_values = [seq[(-offset + y):y] if y < 0 else seq[y:(len(seq) - (1 - (offset + y)))] for y in sorted(y_index,key=lambda x: abs(x))]
    # Get x|y keys
    xys = tuple(product(np.unique(x_values),tuple(product(*[set(y) for y in y_values]))))
    # Get x|y sequences
    xy_seq = zip(x_values,*y_values)
    # Return dict of x|y keys and their probability of occurence in the sequence
    return dict([((x,y),xy_seq.count((x,) + y)/zip(*y_values).count(y)) if zip(*y_values).count(y) else ((x,y),0.0) for x,y in xys])

def _parse_time_window(sequence_length, time_window, subseq_length=0):
    # If negative, then we're creating cumulative subsequences using time_window as a 'memory' value
    if time_window < 0:
        time_window = min(-time_window, sequence_length)       
        time_window = [slice(max(idx - time_window,0),idx) for idx in xrange(subseq_length + 1, sequence_length + 1)]
    # Otherwise, we're breaking the sequence into non-overlapping subsequences
    elif time_window > 0:
        time_window = int(ceil(sequence_length / (1 if time_window > sequence_length else time_window)))
        time_window = [slice(idx,idx + time_window) for idx in xrange(subseq_length, sequence_length, time_window)]
    else:
        raise ValueError("'time_windows' value must be a non-zero integer")
    return time_window

def entropy(probabilities):
    """
    Returns the shannon entropy from the given list of probabilities.

    Args:
        probabilities (list): probabilities of each value

    Returns:
        :float: shannon entropy (bits)
    """
    return -sum([probability * log(probability,2) for probability in probabilities if probability != 0])

def shannon_entropy(probability_dict):
    """
    Returns the shannon entropy from the given probability dictionary.

    Args:
        probability_dict (dict): {x:probability}
            x: x value
            probability: probability of x

    Returns:
        :float: shannon entropy (bits)
    """
    return -sum([probability * log(probability,2) for probability in probability_dict.values() if probability != 0])

# This is a bit of a hack, but I want to allow someone calculating joint entropy to be able to be explicit in their code
# This works because (functionally) shannon and joint entropy are calculated similarly once the probabilities have been determined
joint_entropy = shannon_entropy

def conditional_entropy(conditional_probability_dict, joint_probability_dict):
    """
    Returns the conditional entropy from the given probabilities.

    Args:
        joint_probability_dict (dict): {xy:probability}
            xy: x,y value
            probability: probability of x,y            
        conditional_probability_dict (list): {xy:probability}
            xy: x|y value
            probability: probability of x|y  

    Returns:
        :float: shannon entropy (bits)
    """
    return -sum([joint_probability_dict[xy] * log(conditional_probability_dict[xy],2) for xy in sorted(conditional_probability_dict.keys()) if conditional_probability_dict[xy] != 0])

def mutual_information(x_probability_dict, y_probability_dict, joint_probability_dict):
    """
    Returns the conditional entropy from the given probabilities.

    Args:
        x_probability_dict (dict): {xy:probability}
            x: x value
            probability: probability of x
        y_probability_dict (dict): {xy:probability}
            y: y value
            probability: probability of y            
        joint_probability_dict (dict): {xy:probability}
            xy: x,y value
            probability: probability of x,y 

    Returns:
        :float: mutual information (bits)
    """
    return sum([joint_probability * log((joint_probability/(x_probability_dict[x] * y_probability_dict[y])),2) for (x,y),joint_probability in joint_probability_dict.items() if joint_probability != 0])

def variation_of_information(x_probability_dict, y_probability_dict, joint_probability_dict):
    """
    Returns the variation of information from the given probabilities.

    Args:
        x_probability_dict (dict): {xy:probability}
            x: x value
            probability: probability of x
        y_probability_dict (dict): {xy:probability}
            y: y value
            probability: probability of y            
        joint_probability_dict (dict): {xy:probability}
            xy: x,y value
            probability: probability of x,y 

    Returns:
        :float: variation of information (bits)
    """
    return shannon_entropy(x_probability_dict) + shannon_entropy(y_probability_dict) - (2 * mutual_information(**locals()))

def shannon_entropy_from_sequence(seq, time_window=None):
    """
    Returns the shannon entropy from a sequence. Particular subsequences can be specified using the time_windows argument.

    Args:
        seq (iterable): sequence.
        time_window (int or list): 
            A positive integer indicates the number of non-overlapping time windows to partition the sequence into; if the length of the sequence is not evenly divisible it will print a warning and the final entropy value will be on a shortened window.
            A negative integer indicates the 'memory' as the sequence is traversed; shannon entropy will be calculated at each point in the sequence including the previous time_window number of values. Set this to the length of the sequence for 'infinite' memory.
                Values larger than the sequence length will be reduced to this.
            If a list (or list of lists) is provided, shannon entropy will be calculated only over these indices in the sequence.

    Returns:
    If time_window is None:
        :float: shannon entropy of the sequence
    Otherwise:
        :list: shannon entropy for each subsequence specified
    """
    # If no time window supplied, calculate entropy on the full sequence
    if time_window is None:
        seq_entropy = joint_entropy(_probabilities(seq))
    # Otherwise, determine each subsequence that entropy needs to be calculated on and then recursively call this function on each
    else:
        if isinstance(time_window,int):
            time_window = _parse_time_window(len(seq), time_window)
        # If only a single list of indices was supplied, turn it into a nested list
        elif isinstance(time_window[0], int):
            time_window = [time_window]
        # Calculate shannon entropy on each subsequence
        seq_entropy = [shannon_entropy_from_sequence(seq[window]) for window in time_window]
    return seq_entropy

def joint_entropy_from_sequence(seq , y_index=-1, time_window=None):
    """
    Returns the joint entropy of a sequence.

    Args:
        seq (iterable): sequence.
        y_index (int or list): relative index of y from x.
        time_window (int or list): 
            A positive integer indicates the number of non-overlapping time windows to partition the sequence into; if the length of the sequence is not evenly divisible it will print a warning and the final entropy value will be on a shortened window.
            A negative integer indicates the 'memory' as the sequence is traversed; joint entropy will be calculated on the subsequences from each point in the sequence including the previous time_window number of values. Set this to the length of the sequence for 'infinite' memory.
                Values larger than the sequence length will be reduced to this.
            If a list (or list of lists) is provided, joint entropy will be calculated only for subsequences within these indices in the sequence.

    Returns:
    If time_window is None:
        :float: joint entropy of the sequence
    Otherwise:
        :list: joint entropy for each subsequence specified
    """
    # If no time window supplied, calculate entropy on the full sequence
    if time_window is None:
        seq_entropy = joint_entropy(_joint_probabilities(seq, y_index))
    # Otherwise, determine each subsequence that entropy needs to be calculated on and then recursively call this function on each
    else:
        if isinstance(time_window,int):
            time_window = _parse_time_window(len(seq), time_window, y_index)
        elif isinstance(time_window[0], int):
            time_window = [time_window]
        seq_entropy = [joint_entropy_from_sequence(seq[window],y_index) for window in time_window] 
    return seq_entropy

def conditional_entropy_from_sequence(seq , y_index=-1, time_window=None):
    """
    Returns the conditional entropy of a sequence.

    Args:
        seq (iterable): sequence.
        y_index (int or list): relative index of y from x.
        time_window (int or list): 
            A positive integer indicates the number of non-overlapping time windows to partition the sequence into; if the length of the sequence is not evenly divisible it will print a warning and the final entropy value will be on a shortened window.
            A negative integer indicates the 'memory' as the sequence is traversed; joint entropy will be calculated on the subsequences from each point in the sequence including the previous time_window number of values. Set this to the length of the sequence for 'infinite' memory.
                Values larger than the sequence length will be reduced to this.
            If a list (or list of lists) is provided, joint entropy will be calculated only for subsequences within these indices in the sequence.

    Returns:
    If time_window is None:
        :float: joint entropy of the sequence
    Otherwise:
        :list: joint entropy for each subsequence specified
    """
    # If no time window supplied, calculate entropy on the full sequence
    if time_window is None:
        seq_entropy = conditional_entropy(_conditional_probabilities(seq, y_index), _joint_probabilities(seq, y_index))
    # Otherwise, determine each subsequence that entropy needs to be calculated on and then recursively call this function on each
    else:
        if isinstance(time_window,int):
            time_window = _parse_time_window(len(seq), time_window, y_index)
        elif isinstance(time_window[0], int):
            time_window = [time_window]
        seq_entropy = [conditional_entropy_from_sequence(seq[window],y_index) for window in time_window] 
    return seq_entropy

def mutual_information_from_sequence(seq , y_index=-1, time_window=None):
    """
    Returns the mutual information of a sequence.

    Args:
        seq (iterable): sequence.
        y_index (int or list): relative index of y from x.
        time_window (int or list): 
            A positive integer indicates the number of non-overlapping time windows to partition the sequence into; if the length of the sequence is not evenly divisible it will print a warning and the final entropy value will be on a shortened window.
            A negative integer indicates the 'memory' as the sequence is traversed; mutual information will be calculated on the subsequences from each point in the sequence including the previous time_window number of values. Set this to the length of the sequence for 'infinite' memory.
                Values larger than the sequence length will be reduced to this.
            If a list (or list of lists) is provided, mutual information will be calculated only for subsequences within these indices in the sequence.

    Returns:
    If time_window is None:
        :float: mutual information of the sequence
    Otherwise:
        :list: mutual information for each subsequence specified
    """
    # If no time window supplied, calculate entropy on the full sequence
    if time_window is None:
        # If y_index is int, convert to list so it can be handled using comprehension
        if isinstance(y_index, int):
            y_index = [y_index]
        # Max of y indices will determine how much the sequence needs to be truncated for x and other y indices
        offset = max(y_index,key=lambda x: abs(x))
        # Get y values of sequence
        y_values = zip(*[seq[(-offset + y):y] if y < 0 else seq[y:(len(seq) - (1 - (offset + y)))] for y in sorted(y_index,key=lambda x: abs(x))])
        seq_mi = mutual_information(_probabilities(seq[-offset:] if offset < 0 else seq[:-offset]), _probabilities(y_values), _joint_probabilities(seq, y_index))
    else:
        if isinstance(time_window,int):
            time_window = _parse_time_window(len(seq), time_window, y_index)
        elif isinstance(time_window[0], int):
            time_window = [time_window]
        seq_mi = [mutual_information_from_sequence(seq[window],y_index) for window in time_window] 
    return seq_mi

def variation_of_information_from_sequence(seq , y_index=-1, time_window=None):
    """
    Returns the variation of information of a sequence.

    Args:
        seq (iterable): sequence.
        y_index (int or list): relative index of y from x.
        time_window (int or list): 
            A positive integer indicates the number of non-overlapping time windows to partition the sequence into; if the length of the sequence is not evenly divisible it will print a warning and the final entropy value will be on a shortened window.
            A negative integer indicates the 'memory' as the sequence is traversed; mutual information will be calculated on the subsequences from each point in the sequence including the previous time_window number of values. Set this to the length of the sequence for 'infinite' memory.
                Values larger than the sequence length will be reduced to this.
            If a list (or list of lists) is provided, mutual information will be calculated only for subsequences within these indices in the sequence.

    Returns:
    If time_window is None:
        :float: variation of information of the sequence
    Otherwise:
        :list: variation of information for each subsequence specified
    """
    # If no time window supplied, calculate entropy on the full sequence
    if time_window is None:
        # If y_index is int, convert to list so it can be handled using comprehension
        if isinstance(y_index, int):
            y_index = [y_index]
        # Max of y indices will determine how much the sequence needs to be truncated for x and other y indices
        offset = max(y_index,key=lambda x: abs(x))
        # Get y values of sequence
        y_values = zip(*[seq[(-offset + y):y] if y < 0 else seq[y:(len(seq) - (1 - (offset + y)))] for y in sorted(y_index,key=lambda x: abs(x))])
        seq_voi = variation_of_information(_probabilities(seq[-offset:] if offset < 0 else seq[:-offset]), _probabilities(y_values), _joint_probabilities(seq, y_index))
    else:
        if isinstance(time_window,int):
            time_window = _parse_time_window(len(seq), time_window, y_index)
        elif isinstance(time_window[0], int):
            time_window = [time_window]
        seq_voi = [variation_of_information_from_sequence(seq[window],y_index) for window in time_window] 
    return seq_voi

def estimated_entropy(seq, type='shannon', memory=float('inf'), prior=None):
    # TO DO
    pass

def estimated_surprise(seq, type='shannon',  memory=float('inf'), prior=None):
    # TO DO
    pass

def sample_entropy(seq, m=2, r=0.2, r_ratio=True, normalise_seq=True, distance_metric='Heaviside', delta=1, return_counts=False):
    """
    Calculate the sample entropy of a series of continuous data.

    N.B. Sample entropy is a relative measure of entropy.

    Args:
        seq (iterable): sequence
        m (int): matching template length (default = 2)
        r (int or float): tolerance for matching (default = 0.2)
        r_ratio (bool): whether r should be derived from the SD of the sequence or as an absolute value
        type (str): type of distance metric to use when calculating sample entropy. Allowed values include 'Heaviside', 'Sigmoid', 'Fuzzy', and 'FuzzyM' (default = 'Sample')
            'Sample': standard sample entropy using a Heaviside distance metric as outlined in Costa et al. (2002, 2005)
            'Sigmoid': sample entropy using a Sigmoid local membership function [1/(1 + exp((d - 0.5)/r))] distance metric as outlined in Xie et al. (2008)
            'Fuzzy': sample entropy using a Gaussian fuzzy local membership function [exp(-d**n/r)] distance metric as outlined in Chen et al. (2007)
            'FuzzyM': sample entropy using a Gaussian fuzzy local and global membership function [exp(-d**n/r)] distance metric as outlined in Liu et al. (2013)
        normalise (bool): whether to normalise the sequence first (default=True); N.B. some metrics (e.g., Sigmoid) may require normalisation
        delta (int): step size when constructing sub-sequences (default=1); only changed when computing modified mutli-scale entropy (Wu et al., 2013)
        return_counts (bool): return a tuple (m+1,m) of the similarity counts instead of the sample entropy (default=False); only used when calculating composite mutli-scale entropy (Wu et al., 2014)

    Returns:
        :float: sample entropy
    """
    FUZZY_N = 2
    FUZZYM_N_LOCAL = 3
    FUZZYM_N_GLOBAL = 2

    #def _get_subseqs(_m, baseline=None):
    #    subseqs = np.array([seq[i:i + (_m * delta):delta] for i in xrange(len(seq) - (m * delta))])
    #    if baseline is not None:
    #        if remove_baseline == 'local':
    #            subseqs = np.apply_along_axis(lambda x: x-np.mean(x),-1,subseqs)
    #        elif remove_baseline == 'global':
    #            subseqs = subseqs - np.mean(seq)
    #    return subseqs

    def _max_dist(seq_i, seq_j):
        return max([abs(i - j) for i,j in zip(seq_i,seq_j)])

    #def _similarity_counts(sub_seqs):
    #    return sum([sum([distance_function(_max_dist(i,j)) for j_idx,j in enumerate(sub_seqs[i_idx + delta:])]) for i_idx,i in enumerate(sub_seqs)])

    def _similarity_counts(_m, baseline=None):      
        subseqs = [seq[i:i + (_m * delta):delta] for i in xrange(len(seq) - (m * delta))]
        if baseline is not None:
            if baseline == 'local':
                subseqs = np.apply_along_axis(lambda x: x-np.mean(x),-1,subseqs)
            elif baseline == 'global':
                subseqs = subseqs - np.mean(seq)
        return sum([sum([distance_function(_max_dist(i,j)) for j_idx,j in enumerate(subseqs[i_idx + delta:])]) for i_idx,i in enumerate(subseqs)])


    def _return_sampen(m_count,mplusone_count):
        try:
            return log(m_count / mplusone_count)
        except ValueError:
            return float('inf')
        except ZeroDivisionError:
            return None

    def _return_counts(m_count,mplusone_count):
        return (m_count,mplusone_count)

    n = None

    seq = np.array(seq, dtype='float64')
    if normalise_seq:
        #seq = (seq - seq.min())/(seq.max() - seq.min())
        seq = (seq - np.mean(seq))/np.std(seq)
    if r_ratio:
        r = np.std(seq) * r
    distance_metric = distance_metric.lower()
    distance_function = None
    return_function = _return_counts if return_counts else _return_sampen
    if distance_metric == 'heaviside':
        distance_function = lambda d: d <= r
        #return return_function(_similarity_counts(_get_subseqs(m)), _similarity_counts(_get_subseqs(m + 1)))
        return return_function(_similarity_counts(m), _similarity_counts(m + 1))
    elif distance_metric == 'sigmoid':
        distance_function = lambda d: 1/(1 + exp((d - 0.5)/r))
        #return return_function(_similarity_counts(_get_subseqs(m, 'local')), _similarity_counts(_get_subseqs(m + 1, 'local')))
        return return_function(_similarity_counts(m, 'local'), _similarity_counts(m + 1, 'local'))
    elif distance_metric == 'fuzzy':
        distance_function = lambda d: exp(-((d**FUZZY_N)/r))
        #return return_function(_similarity_counts(_get_subseqs(m, 'local')), _similarity_counts(_get_subseqs(m + 1, 'local')))
        return return_function(_similarity_counts(m, 'local'), _similarity_counts(m + 1, 'local'))
    elif distance_metric == 'fuzzym':
        distance_function = lambda d: exp(-((d**FUZZYM_N_LOCAL)/r)) 
        #local_fuzzym = return_function(_similarity_counts(_get_subseqs(m, 'local')), _similarity_counts(_get_subseqs(m + 1, 'local')))
        local_fuzzym = return_function(_similarity_counts(m, 'local'), _similarity_counts(m + 1, 'local'))
        distance_function = lambda d: exp(-((d**FUZZYM_N_GLOBAL)/r))
        #global_fuzzym =  return_function(_similarity_counts(_get_subseqs(m, 'global')), _similarity_counts(_get_subseqs(m + 1, 'global')))
        global_fuzzym = return_function(_similarity_counts(m, 'global'), _similarity_counts(m + 1, 'global'))
        return tuple(sum(x) for x in zip(local_fuzzym,global_fuzzym)) if return_counts else local_fuzzym + global_fuzzym
    else:
        raise ValueError('Unrecognised distance metric specified.')

def multiscale_entropy(orig_seq, m=2, r=0.2, tau_list=[1], mse_type='MSE', r_rescale=False, **kwargs):
    """
    Calculate the sample entropy of a series of continuous data across a range of temporal delays.
    
    Args:
        seq (iterable): sequence
        tau_list (list of ints): list of temporal delays
        mse_type (str): type of multiscale entropy. Allowed values include 'MSE', 'RMSE', 'MMSE', 'CMSE', and 'RCMSE' (default = 'MSE')
            'MSE': Multi-Scale Entropy as outlined in Costa et al. (2002, 2005); a coarse-grained averaging procedure is used
            'RMSE': Refined Multi-Scale Entropy as outlined in Valencia et al. (2009); a bidirectional 6th-order lowpass Butterworth-filter is used, and r is recaculated for each temporal delay (if r_ratio is also True)
            'MMSE': Modified Multi-Scale Entropy as outlined in Wu et al. (2013b); a moving-average procedure is used, and a delta term equal to tau is used when calculating sample entropy
            'CMSE': Composite Multi-Scale Entropy as outlined in Wu et al. (2013a); a composite coarse-grained/moving average procedure is used
            'RCMSE': Refined Composite Multi-Scale Entropy as outlined in Wu et al. (2014); variation of CMSE which sums the counts over the composite window before calculating SampEn
        r_rescale (bool): when//how to apply r ratio (default = False; N.B. setting mse_type to 'RMSE' will override this setting to True); only used if r_ratio is True
            False: r is used as a proportion of the SD of the original sequence
            True: r is used as a proportion of the SD of the filtered sequence (see Valencia et al., 2009)
        kwargs: additional keyword arugments to pass onto the sample_entropy function (see sample_entropy for details)

    Returns:
        :list: Sample entropy (float) for each temporal delay specified in tau_list
    """
    BUTERWORTH_FILTER_ORDER = 6

    mse_type = mse_type.lower()

    def _butter_lowpass(seq, cutoff_freq):
        nyq = 0.5
        normal_cutoff = cutoff_freq / nyq
        b,a = butter(BUTERWORTH_FILTER_ORDER, normal_cutoff, 'low', analog=False)
        return filtfilt(b, a, seq)

    mse = [None] * len(tau_list)
    kwargs.update({'return_counts':True if mse_type == 'rcmse' else False})
    for idx,tau in enumerate(tau_list):
        kwargs.update({'delta':tau if mse_type == 'mmse' else 1})

        if tau > 1:
            if mse_type == 'mse':
                seq = [np.mean(orig_seq[i:i+tau]) for i in xrange(0,(len(orig_seq) // tau) * tau, tau)]
            elif mse_type == 'mmse':
                seq = [np.mean(orig_seq[i:i+tau]) for i in xrange(len(orig_seq) - (tau - 1))]
            elif mse_type == 'rmse':
                r_rescale = True
                seq = _butter_lowpass(orig_seq, 0.5 / tau)
            elif mse_type in {'cmse','rcmse'}:
                pass # Deal with this below
            else:
                raise ValueError('Unrecognised filter specified.')
        else:
            seq = orig_seq[:]

        if mse_type == 'cmse' and tau > 1:
            mse[idx] = np.mean([sample_entropy([np.mean(orig_seq[i+t:i+t+tau]) for i in xrange(0,(len(orig_seq) // tau) * tau, tau)],m,r,**kwargs) for t in xrange(tau)])
        elif mse_type == 'rcmse' and tau > 1:
            a,b = np.mean([sample_entropy([np.mean(orig_seq[i+t:i+t+tau]) for i in xrange(0,(len(orig_seq) // tau) * tau, tau)],m,r,**kwargs) for t in xrange(tau)],axis=0)
            mse[idx] = log(a/b)
        else:
            mse[idx] = sample_entropy(seq,**kwargs)
    return mse