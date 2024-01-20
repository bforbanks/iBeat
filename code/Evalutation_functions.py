import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import random

# Used for other functions
def convert_to_lists(dictA,dictB):
    listA = []
    listB = []
    for key in dictA:
        listA.append(dictA[key])
        listB.append(dictB[key])
    return listA, listB

# Only used for other functions in this file
def Kulback_Leibler_One(beatsGlobal, annsGlobal, show_hist=False, num_bins=50, switched=False):
    if type(beatsGlobal) == dict and type(annsGlobal) == dict:
        beatsGlobal, annsGlobal = convert_to_lists(beatsGlobal, annsGlobal)
    elif type(beatsGlobal) != list and type(annsGlobal) != list:
        raise Exception(f"Wrong input types, beats: {type(beats)} and annsGlobal: {type(annsGlobal)}, should both be lists or dicts")
    g_beat_errors = [] 
    for snipno in range(len(beatsGlobal)):
        # Check if the snippet has annotations
        if len(annsGlobal[snipno]) < 2:
            print("Snippet number", snipno, "has no annotations. The algorithm will skip this song, note that this will make the final result less representative")
            continue
        # Add extra annotations for the sake of the algorithm, also unpack global list to just one snippet
        mod_anns = [2*annsGlobal[snipno][0]-annsGlobal[snipno][1]]+annsGlobal[snipno]+[2*annsGlobal[snipno][-1]-annsGlobal[snipno][-2]]
        beats=beatsGlobal[snipno]

        # Find the intervals related to each annotation
        ann_intervals = [((mod_anns[i]+mod_anns[i-1])/2) for i in range(1,len(mod_anns))]

        # Find the beat errors for each snippet
        beat_errors=[]
        for i_low in range(len(ann_intervals)-1):

            # Find all the beats within the interval of the current annotation which is mod_anns[i_low+1] (because I added the extra annotation in the beginning)
            beats_for_interval=[]
            for beat in beats:
                if ann_intervals[i_low] <= beat:
                    if ann_intervals[i_low+1] > beat:
                        beats_for_interval.append(beat)
                    else:
                        break

            # Find the beat errors for the beats within the interval and append them to beat_errors
            for beat in beats_for_interval:
                if beat > mod_anns[i_low+1]:
                    a=(beat-mod_anns[i_low+1])/(mod_anns[i_low+2]-mod_anns[i_low+1])
                    beat_errors.append(a)
                else:
                    a=(beat-mod_anns[i_low+1])/(mod_anns[i_low+1]-mod_anns[i_low])
                    beat_errors.append(a)
        
        # Extend the global (accross snippets) beat errors list
        g_beat_errors.extend(beat_errors)

    
    # Plot the histogram of g_beat_errors, if specified in function call
    if show_hist:
        plt.hist(g_beat_errors, bins=num_bins, density=True)
        if not switched:
            plt.xlabel('Beat Errors')
            plt.ylabel('Probability')
            plt.title('Probability Histogram of Beat Errors')
        else:
            plt.xlabel('Annotation Errors')
            plt.ylabel('Probability')
            plt.title('Probability Histogram of Annotation Errors')
        plt.xlim([-0.6, 0.6]) 
        plt.show()

    return g_beat_errors


# The main KLd function
def Kulback_Leibler(beats, anns, show_hist=False, num_bins=50, conf_int = True, sign_level = 0.05, sim_count=1000):
    """
    Calculate the Kulback-Leibler divergence metric for a set of songs

    Args::
        beats: list of lists or dictionary with lists of predicted beats
        anns: list of lists or dictionary with lists of correct annotations
        show_hist: show the two beat errors histograms
        num_bins: number of bins for the histograms
        conf_int: calculate the confidence interval
        sign_level: significance level for the confidence interval
        sim_count: number of simulations for the confidence interval

    Returns:
        The KLd value and the confidence interval in a tuple if specified
        (KLd, (lower_bound, upper_bound)) if conf_int is True
        """
    u_beat_errs = Kulback_Leibler_One(beats, anns, show_hist, num_bins, False)
    s_beat_errs = Kulback_Leibler_One(anns, beats, show_hist, num_bins, True)
    if conf_int:
        sim_values=[]
        for i in range(sim_count):
            unswitched_beats_sim = random.choices(u_beat_errs, k=len(u_beat_errs))
            switched_beats_sim = random.choices(s_beat_errs, k=len(u_beat_errs))
            hist_unswitched, _ = np.histogram(unswitched_beats_sim, bins=num_bins, density=True)
            hist_switched, _ = np.histogram(switched_beats_sim, bins=num_bins, density=True)
            unswitched_value = np.log2(num_bins)-sum([i/num_bins*np.log2(1/(i/num_bins)) for i in hist_unswitched if i != 0])
            switched_value = np.log2(num_bins)-sum([i/num_bins*np.log2(1/(i/num_bins)) for i in hist_switched if i != 0])
            sim_values.append(min(unswitched_value, switched_value))

        lower_bound = np.quantile(sim_values, sign_level/2)
        upper_bound = np.quantile(sim_values, 1-sign_level/2)
        hist_unswitched, _ = np.histogram(u_beat_errs, bins=num_bins, density=True)
        hist_switched, _ = np.histogram(s_beat_errs, bins=num_bins, density=True)
        unswitched_value = np.log2(num_bins)-sum([i/num_bins*np.log2(1/(i/num_bins)) for i in hist_unswitched if i != 0])
        switched_value = np.log2(num_bins)-sum([i/num_bins*np.log2(1/(i/num_bins)) for i in hist_switched if i != 0])
        best_guess = min(unswitched_value, switched_value)
        return best_guess, (lower_bound, upper_bound)
    else:
        hist_unswitched, _ = np.histogram(u_beat_errs, bins=num_bins, density=True)
        hist_switched, _ = np.histogram(s_beat_errs, bins=num_bins, density=True)
        unswitched_value = np.log2(num_bins)-sum([i/num_bins*np.log2(1/(i/num_bins)) for i in hist_unswitched if i != 0])
        switched_value = np.log2(num_bins)-sum([i/num_bins*np.log2(1/(i/num_bins)) for i in hist_switched if i != 0])
        return min(unswitched_value, switched_value)

# The main F_measure function
def F_measure(beatsGlobal, annsGlobal, error=25, mode="single", return_mean=True):
    """
    Calculates the F_measure for a set of songs

    Args:
        beatsGlobal: list of lists or dictionary with lists predicted beats
        annsGlobal: list of lists or dictionary with lists of correct annotations
        error: the margin of error allowed on both sides of annotation
        mode: single or global, single computes f-measure for each song, global computes f-measure for all songs combined
        return_mean: Return mean instead of list of F-values

    Returns:
        The F_measure value or list of F-values for each song if mode is single and return_mean is False
    """
    if type(beatsGlobal) == dict and type(annsGlobal) == dict:
        beatsGlobal, annsGlobal = convert_to_lists(beatsGlobal, annsGlobal)
    elif type(beatsGlobal) != list and type(annsGlobal) != list:
        print("Wrong input types")
    correct = 0
    false_positive=0
    false_negative=0
    F_values=[]
    for i in range(len(beatsGlobal)):
        if mode=="single":
            correct = 0
            false_positive=0
            false_negative=0
        anns=annsGlobal[i].copy()
        beats=beatsGlobal[i].copy()
        for ann in anns:
            beats_within_window=[]
            for beat in beats.copy():
                if ann-error <= beat and beat <= ann+error:
                    beats_within_window.append(beat)
                    beats.remove(beat)
            # print(round(ann), [round(num) for num in beats_within_window])
            if len(beats_within_window)!=0:
                false_positive+=len(beats_within_window)-1
                correct+=1
            else:
                false_negative+=1
        false_positive+=len(beats)
        
        # print(beats)
        
        if mode=="single":
            #"Precision indicates the proportion of the generated beats which are correct"
            if correct+false_positive != 0:
                p=correct/(correct+false_positive)
            else:
                p=0
            #"Recall indicates the proportion of the total number of correct beats that were found"
            if correct+false_negative != 0:
                r=correct/(correct+false_negative)
            else:
                r=0
            #"When combined they provide the F-measure accuracy value"
            if (p+r) != 0:
                F=2*p*r/(p+r)
            else:
                F=0
            F_values.append(F)
    if mode=="single":
        if return_mean:
            return np.mean(F_values)
        else:
            return F_values
    else:
        if correct+false_positive != 0:
            p=correct/(correct+false_positive)
        else:
            p=0
        if correct+false_negative != 0:
            r=correct/(correct+false_negative)
        else:
            r=0
        if (p+r) != 0:
                F=2*p*r/(p+r)
        else:
                F=0
        return F