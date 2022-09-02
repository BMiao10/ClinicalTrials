
# system
import os
from pathlib import Path
import glob
import re
from collections import OrderedDict
from collections import Counter
from datetime import datetime
import string

# data
import pandas as pd
import numpy as np
import json
import requests
import pgeocode

# math
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import spearmanr
import math
import scipy
from umap import UMAP

# nlp
import spacy
import scispacy
from scispacy.linking import EntityLinker
from bertopic import BERTopic
#from scispacy.umls_linking import UmlsEntityLinker
#import spacy_streamlit

# plotting
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from statannot import add_stat_annotation
import plotly.express as px
import plotly.graph_objects as go

######################################################
### DTx manuscript functions

def createSupplementalTopicTables(topics_per_class, topics=[0,1,2,3,4], topic_names=None):
    """
    Formats BERTopic values for manuscript table
    
    Params:
        topics_per_class
    
    Returns:
        pd.DataFrame()
    """
    # filter values
    table_df=pd.DataFrame()
    topics_per_class= topics_per_class[topics_per_class["Topic"].isin(topics)]
    topics_per_class = topics_per_class.groupby(["Name", "Class", "Topic"]).first()
    
    # add proportion components
    table_df.index = topics_per_class.index
    table_df["Proportion"] = [ "%s/%s (%.1f%%)"%(f,t,p*100) for f, t, p in zip(topics_per_class["Frequency"], topics_per_class["Total"], topics_per_class["Percent"])]
    table_df["Components"] = topics_per_class["Words"]
    
    # sort values
    table_df["Percent"] = topics_per_class["Percent"]
    table_df = table_df.reset_index()
    #topics_per_class = topics_per_class.sort_values(["Name", "Percent"], ascending=False)
    table_df= table_df.sort_values(["Name", "Percent"], ascending=[True, False])
    del table_df["Percent"]

    # formatting
    if topic_names is not None:
        table_df["Topic names"] = [topic_names[str(n)] for n in table_df["Topic"]]
    
        del table_df["Topic"]
        table_df.columns = ["Global topic representation", "MeSH Heading", "Proportion", "Components", "Topic names"]
        table_df = table_df[["Topic names", "MeSH Heading", "Proportion", "Components"]]
    else:
        del table_df["Topic"]
        table_df.columns = ["Topic names", "MeSH Heading", "Proportion", "Components"]
        table_df = table_df[["Topic names", "MeSH Heading", "Proportion", "Components"]]
    
    return table_df.set_index(["Topic names", "MeSH Heading"])
     

def printStudyMetrics(metrics_df, groupby="StudyType", study_type="Interventional", 
                      values = "NumberLocations", metrics=["count", "median", "std"]):
    """
    Prints metrics for values in a specified group 
    
    Params:
        metrics_df
        groupby (str): col to groupby
        study_type (str): group to print metrics for
        values (str): value column
        metrics (list): type of metrics to print (must be list of 3 values)
    
    Returns:
        None
    """
    int_count, int_mean, int_std = metrics_df.groupby(groupby)[[values]].agg(metrics).loc[study_type][values]
    metrics_df = metrics_df[metrics_df[groupby]==study_type]
    median = metrics_df[values].median()
    iqr_25, iqr_75 = np.percentile(metrics_df[values], q=[25 , 75], interpolation="midpoint")
    
    print("%s %s"%(study_type, values))
    print("Count: %s"%int_count)
    print("%s: %.2f "%(metrics[1], int_mean))
    print("IQR (25%%), IQR (75%%): %.2f-%.2f"%(iqr_25, iqr_75))
    print("Range: %.2f-%.2f"%(metrics_df[values].min(), metrics_df[values].max()))
    #print("95%% CI: %.2f - %.2f"%((int_mean - 1.96*int_std), (int_mean + 1.96*int_std)))
    print()

def printEnrollmentMetrics(enrollment_df, values=["Actual", "Anticipated"]):
    """
    Prints count, mean/median, range information for each clinical trial
    
    Params:
        enrollment (list<int, float>): list of enrollment counts (or log counts) 
    """
    for x in enrollment_df["conditionMeshMainBranch"].unique():
        curr_df = enrollment_df[enrollment_df["conditionMeshMainBranch"]==x]

        if values is None:
            print(x+", actual + anticipated")
            enrollment = curr_df["EnrollmentCount"]
            median = enrollment.median()
            mean = enrollment.mean()
            iqr_25, iqr_75 = np.percentile(enrollment, q=[25 , 75], interpolation="midpoint")

            print("Count: %d "%len(enrollment))
            print("Mean, median: %d, %d"%(enrollment.mean(), enrollment.median()))
            print("IQR (25%%), IQR (75%%): %s - %s"%(iqr_25, iqr_75))
            print("Range: %d - %d"%(enrollment.min(), enrollment.max()))
            #print("95%% CI, mean: %.2f - %.2f"%((mean - 1.96*enrollment.std()), (mean + 1.96*enrollment.std())))
            print()
        else: 
            for a in values:
                print(x+", "+a)

                enrollment = curr_df[curr_df["EnrollmentType"] == a]["EnrollmentCount"]
                median = enrollment.median()
                mean = enrollment.mean()
                iqr_25, iqr_75 = np.percentile(enrollment, q=[25 , 75], interpolation="midpoint")

                print("Count: %d "%len(enrollment))
                print("Mean, median: %d, %d"%(enrollment.mean(), enrollment.median()))
                print("IQR (25%%), IQR (75%%): %s - %s"%(iqr_25, iqr_75))
                print("Range: %d - %d"%(enrollment.min(), enrollment.max()))
                #print("95%% CI, mean: %.2f - %.2f"%((mean - 1.96*enrollment.std()), (mean + 1.96*enrollment.std())))
                print()
    

def printTopicValues(topics_per_class, topics=[0,1,2,3,4], sort_col="Percent",
                     freq_col="Frequency", total_col="Total", pct_col="Percent"):
    """
    Formatted printing for topic values
    
    Params:
        topics_per_class (pd.DataFrame): frequency, total, percent values for each class topic
        topics (list<int>): list of topics to print
        sort_col (str): column to sort values for printing
        freq_col (list<str>): list of values to print
    
    """
    
    # filtering
    if topics is not None: 
        topics_per_class = topics_per_class[topics_per_class["Topic"].isin(topics)]
    
    # print percentages for each topic 
    for topic in topics_per_class["Topic"].unique():
        curr_df = topics_per_class[topics_per_class["Topic"] == topic]
        curr_df = curr_df.sort_values(sort_col, ascending=False)
        curr_df = curr_df.set_index("Class")
        print(curr_df["Name"].unique()[0])
        for c in curr_df.index: 
            row = curr_df.loc[c]
            print("\t%s: %s/%s (%.2f)"%(c, row[freq_col], row[total_col], row[pct_col]*100), row["Words"])
        print()
        

def plotPopulationvsNumTrialLocations(state_df, min_label=25, height=8):
    """
    Scatterplot of state population vs number of clinical trials
    """
    state_df = state_df.dropna(subset=["state_code"])
    state_df["clinicalTrials"] = state_df["clinicalTrials"].replace(np.nan, 0)
    sns.set(font_scale = 2)
    sns.set_style("white")
    sns.set_context("talk")
    lm = sns.lmplot(data = state_df, x = "July2021Estimate", y= "clinicalTrials", height=height)
    ax = lm.axes[0,0]
    print("State population vs # trial sites: ", spearmanr(state_df["clinicalTrials"], state_df["July2021Estimate"]))

    labels = state_df.copy(deep=True)
    labels["label"] = labels["state_code"]
    labels = labels[labels["clinicalTrials"] > min_label]
    ax = addLabels(ax, labels=labels, xValues="July2021Estimate", yValues="clinicalTrials", 
                   plotType='scatter', size=16, color='black', correction=1.5)

    ax.set_ylabel("Number of DTx trial locations")
    ax.set_xlabel("State population (100k)")
    
    return ax  

def plotMissingness(missing_df, sort_values="BaselineGroup", max_values=0.7, 
                    figsize=(12,18), row_colors = "StudyType", **kwargs):
    """
    Plots heatmap of missingness, with values sorted by specified columns
    """
    total = missing_df.shape[0]
    curr_df = missing_df.copy(deep=True)
    max_values = int(total*max_values)
    min_col = None
    
    # convert to missingness
    for i in missing_df.columns:
        curr_df[i] = [0 if x is None else 1 for x in curr_df[i]]
        
        if sum(curr_df[i]) <= max_values:
            min_col = i
            min_value = sum(curr_df[i])
        
        if sum(curr_df[i]) > max_values:
            del curr_df[i]
    
    curr_df = curr_df.sort_values(sort_values)
    
    # plot
    g = sns.clustermap(curr_df.T, figsize=figsize, dendrogram_ratio=0.1, cbar_pos=None, xticklabels=False, yticklabels=1)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 14)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.yticks(ha='left',) 
    plt.tick_params(length=1)

    return g

def plotTrialLocationsPerADI(state_df, rank_type="ADI_NATRANK", height=8):
    """
    Plots number of locations per ADI
    
    Params:
        rank_type (str): "ADI_NATRANK" or "ADI_STATERNK," see University of Wisconsin ADI dataset for more info
    """
    # get all clincal trial location values
    state_df = state_df.explode(rank_type)
    
    # group and plot values (national)
    state_df = state_df.groupby(rank_type).count().reset_index()
    state_df[rank_type] = state_df[rank_type].astype(int, errors="ignore")
    
    lm = sns.lmplot(data = state_df, x = rank_type, y= "NCTId", height=height)
    lm.set(ylim=(0, None))
    ax = lm.axes[0,0]
    ax.set_ylabel("Number of DTx trial locations")
    print("# trial sites vs ADI: ", spearmanr(state_df["NCTId"], state_df[rank_type]))
    
    return ax

def getLocationZipADI(state_df, states=["CA", "TX", "FL", "NY", "PA"]):
    """
    Gets ADI for each clinical trial zip code
    Can only do a few states because the ADI zip code mapping files are pretty big and can only be downloaded state by state
    2019 ADI data pulled from here: https://www.neighborhoodatlas.medicine.wisc.edu/download
    Last accessed: August 29, 2022
    
    Params:
        state_df (pd.DataFrame): clinical trial dataframe with LocationZip
        states (list<str>): list of 2-letter codes for state of interest
        
    Returns ADI values 
    """
    # load ADI values and map to zip
    state_df = state_df.explode("LocationZip")
    nat_dict = {}
    state_dict = {}
    color_dict = {}
    
    for s in states:
        curr_df = pd.read_csv("./dataInput/State_ADI_Data/%s_2019_ADI_9 Digit Zip Code_v3.1.txt"%s, 
                              sep=",", index_col=0, dtype={'TYPE': 'str', "ADI_NATRANK":"str", 'ADI_STATERNK': 'str'})
        
        #drop missing values
        curr_df = curr_df[curr_df["ADI_NATRANK"]!="GQ"] #GQ = high group quarters population
        curr_df = curr_df[curr_df["ADI_NATRANK"]!="PH"] #PH = low population/housing 
        curr_df = curr_df[curr_df["ADI_NATRANK"]!="GQ-PH"] #PH-GQ = both types of suppression criteria
        curr_df = curr_df[curr_df["ADI_NATRANK"]!="PH-GQ"]
        curr_df = curr_df[curr_df["ADI_NATRANK"]!="KVM"] #KVM = Key Missing Variables
        
        # update values dicts
        curr_df["zipclean"] = [z[1:6] for z in curr_df["ZIPID"]]
        nat_dict.update(dict(zip(curr_df["zipclean"], curr_df["ADI_NATRANK"])))
        state_dict.update(dict(zip(curr_df["zipclean"], curr_df["ADI_STATERNK"])))
    
    # map ADI values to zip
    state_df["ADI_NATRANK"] = state_df["LocationZip"].map(nat_dict)
    state_df["ADI_STATERNK"] = state_df["LocationZip"].map(state_dict)

    return state_df[["NCTId", "ADI_NATRANK", "ADI_STATERNK"]]

        
######################################################
### Helper functions

def addLabels(ax, labels, xValues, yValues, plotType='scatter', size=12, color='black', correction=0.2):
    """
    Add labels to matplotlib plot
    """
    
    for index in labels.index:
        if plotType == 'scatter':
            ax.text(labels[xValues][index]+np.random.uniform(-correction, correction), labels[yValues][index] 
                    +np.random.uniform(-correction, correction),
                    labels['label'][index], horizontalalignment='left', size=size, color=color, weight=500)
        else:
            name = labels['label'][index]
            ax.text(xValues.index(labels['type'][index]) + correction,  labels[yValues][index] + correction, 
                    name, horizontalalignment='left', size=size, color=color, weight=500)
            
    return ax

def mannWhitneyLongDf(stats_df, values_col,  labels_col, subset_col=None, 
                       min_values=10, avg="median", correct=True, sig_only=True):
    """
    Runs statistical test on long form data for two categories, returns median values
    
    Params:
        stats_df (pd.DataFrame): dataframe
        values_col (str): column containing data to compare by mann whitney
        labels_col (str): column containing groups to compare by mann whitney
        subset_col (str, None): column contaning subsets to run statistical testing for
        min_values (str): min values in each subset to run statistical testing for
        avg (str): 'mean' or 'median'
        correct (bool): bonferroni correction on number of subset groups
        sig_only (bool): only return significant groups by mann whitney
        (TODO) labelA, B (str): selection of groups within labels_col
        
    Returns:
        pd.DataFrame: pvalues for mann whitney results
        
    """
    
    stats_dict = {}

    # if subset_col not defined, use all values
    if subset_col is None: 
        stats_df["subset_col"] = 1
        subset_col = "subset_col"
    
    # calculate average and mannwhitney between categories
    for b in stats_df[subset_col].unique():
        curr_df = stats_df[stats_df[subset_col] == b]
        diff = 0
        
        if len(curr_df) >= min_values:
            groupA_label = curr_df[labels_col].unique()[0]
            groupB_label = curr_df[labels_col].unique()[1]
            
            groupA = curr_df[curr_df[labels_col] == groupA_label]
            groupB = curr_df[curr_df[labels_col] == groupB_label]
            
            if (len(groupB[values_col].dropna()) > 3) and (len(groupA[values_col].dropna()) > 3):
                x = mannwhitneyu(groupA[values_col].dropna(), y=groupB[values_col].dropna(), )
                groupA_avg = 0
                groupB_avg = 0

                if avg=="mean":
                    groupA_avg= groupA[values_col].mean()
                    groupB_avg= groupB[values_col].mean()
                elif avg=="median":
                    groupA_avg= groupA[values_col].median()
                    groupB_avg= groupB[values_col].median()

                stats_dict[b]=(x[1], groupA_avg, groupB_avg)
            else:
                print("Not enoughh values for stats on %s"%b)
                
    pvals_df = pd.DataFrame.from_dict(stats_dict, orient="index")
    pvals_df.columns = ["pval", groupA_label+" "+ str(avg), groupB_label+" "+ str(avg)]

    if correct: # bonferroni 
        pvals_df["pval"] = pvals_df["pval"]*len(pvals_df)
    if sig_only:
        pvals_df = pvals_df[pvals_df["pval"]<0.05]
        
    return pvals_df

### Sponsor & trial location cleaning
def collapseCollaboratorType(collab_class, multiple="group"):
    """
    Get primary collabortor type for ClinicalTrials.gov data
    Collapses duplicate values and groups values for multiple collaborator types
    
    Params:
        collab_class (list<list<str>>): list of collaborators for each clinical trial
        multiple (str): how to deal with multiple values, "group", "common", or custom list order
        
    Returns:
        Primary collaborator type for each clinical trial
        If multiple is "group," creates separate Multiple label 
        If multiple is "ignore," returns most common collaborator type
        Can also specify custom order for groupings by setting mutiple to a list with 
        trials assigned to the first-occuring collaborator type prioritized in the list order.
        
    """    
    # if "common", get most common value 
    if multiple == "common": # get most common in each
        return [max(set(c), key = c.count) if type(c)==list else "None" for c in collab_class]
    
    # Otherwise, collapse collaborator and group/sort
    collab_class = ["NONE" if c is None else "NONE" if type(c)==float else set(c) for c in collab_class]

    if multiple == "group":
        return ["NONE" if c=="NONE" else list(c)[0] if len(c) == 1 else "MULTIPLE" for c in collab_class]
    
    elif type(multiple)==list: 
        for m in multiple:
            collab_class = [c if type(c)==str 
                            else m if m in c 
                            else c for c in collab_class]
    
        collab_class = [c if type(c)==str else "OTHER/UNKNOWN" for c in collab_class]
        return collab_class

def getTrialLocations(zip_codes, country="us"):
    """
    Get latitude and longitude values for all locations
    
    Params:
        zip_codes (list<int, list<int>>): zip code values from ClinicalTrials.gov LocationZip field
        country (str): country to search for
        
    Returns:
        None, saves lat/long values to ./dataOutput/zip_lat_lng.csv
    """
    # map values to lat/long using cached values & pgeocode
    # https://pypi.org/project/pgeocode/
    nomi = pgeocode.Nominatim(country)

    # get all zip codes
    all_zips = set(zip_codes.explode())
    all_zips = [str(z) if type(z)!=str 
                else z[:5] if len(z)>5 and "-" in z 
                else z for z in all_zips]

    # read cached values
    zip_lat_lng = pd.read_csv("./dataOutput/zip_lat_lng.csv")
    zip_lat_lng["postal_code"] = zip_lat_lng["postal_code"].astype(str)
    zip_lat_lng = zip_lat_lng.dropna(subset=["latitude"])
    
    # get non-cached values
    cache_lat = dict(zip(zip_lat_lng["postal_code"], zip_lat_lng["latitude"]))

    for z in all_zips:
        if ((str(z) not in cache_lat.keys()) & (len(z)>0)):
            z_lat_lng = nomi.query_postal_code(z)
            zip_lat_lng = zip_lat_lng.append(z_lat_lng.transpose())
        elif math.isnan(cache_lat[z]):
            z_lat_lng = nomi.query_postal_code(z)
            zip_lat_lng = zip_lat_lng.append(z_lat_lng.transpose())
    
    zip_lat_lng = zip_lat_lng.dropna(subset=["latitude"])
    zip_lat_lng = zip_lat_lng.drop_duplicates("postal_code")
    zip_lat_lng = zip_lat_lng.set_index("postal_code", drop=True)
    zip_lat_lng.to_csv("./dataOutput/zip_lat_lng.csv") # update with new values
    
def mapTrialLocations(loc_df, zip_file="./dataOutput/zip_lat_lng.csv"):
    """
    Map latitude/longitude values for each clinical trial
    
    Params:
        loc_df (pd.DataFrame): contains "NCTId" and "LocationZip" values
        zip_file (str, filepath): file containing pgeocode values for each zip code
        
    Returns:
        pd.DataFrame with lat/long and additional pgeocode values for each clinical trial
    """
    zip_lat_lng = pd.read_csv(zip_file)
    
    # map lat/long to studies
    loc_df = loc_df.explode('LocationZip')
    loc_df["LocationZipClean"] = [str(z) if type(z)!=str  
                                  else z[:5] if len(z)>5 and "-" in z 
                                  else z for z in loc_df["LocationZip"]]
    loc_df = loc_df.merge(zip_lat_lng, left_on="LocationZipClean", right_on="postal_code", how="left")
    
    # create lists
    loc_df = loc_df.groupby('NCTId').agg(lambda x: list(x))
    
    return loc_df

######################################################
### Condition data cleaning
def extractConditionMeshBranch(cond_list, **kwargs):
    """
    Given list of clinical trial conditions, map each condition to a MeSH branch
    
    Params:
        cond_list (list, pd.Series): list of clinical trial condition values
        //collapse (bool): if True, returns only most common MeSH branch for each value, otherwise returns full list
        **kwargs: extractMeshEnts
        
    Returns:
        dict<str:list>: dictionary of condition:mapped MeSH branch
    """
    
    all_ents = set(cond_list.sum())
    ent_dict = {}
    
    # get entities for all conditions
    for a in all_ents:
        ent_dict[a] = extractMeshEnts(a, **kwargs)

    # map entities to cond_list
    conditions = []
    for c in cond_list:
        curr_cond = []
        for r in c:
            if r in ent_dict.keys():
                curr_cond.extend(ent_dict[r])
        conditions.append(curr_cond)
    
    return conditions

def extractMeshEnts(text, mesh_dict, _nlp, lemma=False, stopwords=None):
    """
    Extract entities from text -> identify MeSH terms (using EntityLinkers) -> map to MeSH branches
    
    Params:
        text (str): string to extract MeSH Entities for
        _nlp (scispacy model): Scispacy model with EntityLinker 
        mesh_dict (dict): dictionary of Mesh branch values
        
    Returns:
        list: list of mapped mesh branch values
    """
    
    # Previously lowercase but turns out the Spacy EntityLinker is case sensitive 
    #text = text.lower().strip()

    # preprocessing
    if lemma ==True: 
        model = _nlp(text)
        text = " ".join([word.lemma_ for word in model])
        
    if stopwords is not None:
        text = " ".join([t for t in text.split(" ") if t not in stopwords])
        
    # map entities to Mesh terms
    model = _nlp(text)
    mesh_ids = [(m.text, m._.kb_ents[0][0]) if (len(m._.kb_ents)>0) else (m.text, "Unknown") for m in model.ents ]
    
    # return MeSH branch for each value
    mesh_branches = [(curr_ent[0], curr_ent[1], mesh_dict[curr_ent[1]]["branch_name"], 
                      mesh_dict[curr_ent[1]]["main_branch"])  if curr_ent[1] in list(mesh_dict.keys()) 
                     else (curr_ent[0], curr_ent[1], "Unknown", "Unknown") for curr_ent in mesh_ids] 
    
    return list(mesh_branches)

def filterMeshBranches(mesh_ids, branches=["C", "F"], exclude=False):
    """
    Limit Scispacy linked MeSH values to specific branches
    
    Params:
        mesh_ids (list): list of Mesh values mapped by Scispacy EntityLinker 
        branches (list<str>): branch values to limit to, default are disease branches "C" and "F"
    """
    if exclude: 
        return [m for m in mesh_ids if len(m)>0 if all(b not in m[3] for b in branches)]
    
    return [m for m in mesh_ids if len(m)>0 if any(b in m[3] for b in branches)]

def collapseMeshEnts(mesh_ids, prefer=["C", "F"]):
    """
    Get most common MeSH branch from a list of Scispacy mapped Mesh values
    
    Params:
        mesh_ids (list): list of Mesh values mapped by Scispacy EntityLinker 
        prefer (list): Mesh branches to prioritize
    """
    best = [m[2] for m in mesh_ids if any(p in m[3] for p in prefer)]
    best = [m for m in best if "Unknown" not in m]

    # take the most common preferred value
    if len(best) > 2: 
        return max(set(best), key=best.count)
    elif len(best) > 0: 
        return best[0]
    
    # take the most common value
    common = [m[2] for m in mesh_ids if len(m)>0]
    common = [m for m in common if "Unknown" not in m]
    if len(common) > 2: 
        return max(set(common), key=common.count)
    elif len(common) > 0: 
        return common[0]
    
    # return unknown only if all the values are unknown
    return "Unknown"

######################################################
### Eligibility criteria cleaning
def extractInclusionExclusionCriteria(elig_df, **kwargs):
    """
    Extract inclusion and exclusion criteria from EligibilityCriteria columns in clinicalTrials.gov data
    
    Params:
        plots_df (pd.DataFrame): contains ["NCTId", "EligibilityCriteria"]
        nlp (spacy or stanza model): NER-enabled NLP model used to extract criteria 
        extractEnts (func): defines how NERs should be extracted, default uses scispacy EntityLinker module
        **kwargs: passed to extractEnts
        
    Returns:
        pd.DataFrame containing parsed inclusion and exclusion criteria for each clinical trial
    """

    ### Extract inclusion and exclusion criteria
    # extract inclusion criteria
    elig_df = elig_df.set_index("NCTId", drop=True)
    elig_df = elig_df['EligibilityCriteria'].str.split('Inclusion Criteria', expand=True)
    elig_df = elig_df.stack()
    elig_df = elig_df.reset_index()
    elig_df = elig_df[elig_df[0] != ""]
    
    # get exclusion criteria
    elig_df[["InclusionCriteria", "ExclusionCriteria"]] = elig_df[0].str.split('Exclusion Criteria', expand=True, n=1)
    elig_df = elig_df.sort_values("level_1")
    elig_df = elig_df.replace(np.nan, "")
    
    # collapse criteria by NCTId
    inclusion_df = elig_df.groupby(['NCTId'])['InclusionCriteria'].apply(lambda x: ' '.join(x).strip()).reset_index()
    exclusion_df = elig_df.groupby(['NCTId'])['ExclusionCriteria'].apply(lambda x: ' '.join(x).strip()).reset_index()
    
    return inclusion_df.merge(exclusion_df, how="outer",left_on="NCTId", right_on="NCTId")

def extractIndividualEligibility(bert_df, criteria_col="ExclusionCriteria", stopwords=[]):
    """
    Get each line from clincial trial eligibility criteria
    
    Params: 
        bert_df (pd.DataFrame): clinical trials data with inclusion and exclusion criteria split
        criteria_col (str): criteria column in bert_df
        
    Returns: 
    
    """
    # explode criteria to individual lines
    embed_col = criteria_col+"Embed"
    bert_df[embed_col] = [n.split("\n") for n in bert_df[criteria_col]]
    bert_df[embed_col] = [[k.strip() for k in c if len(k)>1] for c in bert_df[embed_col]]
    bert_df = bert_df.explode([embed_col])

    # clean text values
    bert_df["%sClean"%embed_col] = cleanTextValues(bert_df[embed_col], regex='[^\sA-Za-z0-9 ><≤≥\.]+', stopwords=stopwords)
    bert_df["%sClean"%embed_col] = [b.replace(" - ", "-") for b in bert_df["%sClean"%embed_col]]
    bert_df = bert_df[bert_df["%sClean"%embed_col] != ""]
    bert_df = bert_df.dropna(subset=["%sClean"%embed_col])
    
    # lemma - the problem with this is it removes things like ios and is not great on the clinical text side
    # tried NLTK stemming as well, but that removed a lot more suffixes that it wasn't supposed to
    # bert_df["InclusionCriteriaEmbedClean"] = [" ".join([tok.lemma_ if len(tok.text)>3 else tok.text for tok in spacy_nlp(c)]) for c in bert_df["InclusionCriteriaEmbedClean"]]

    
    return bert_df

#@st.experimental_singleton
def extractBERTopics(bert_df, _nlp, seed=None, nr_topics='auto',
                      criteria_col = "ExclusionCriteriaEmbedClean", class_col='conditionMeshMainBranch'):
    """
    Get topics using BERTopic run on spacy embeddings 
    https://arxiv.org/abs/2203.05794
    
    Params: 
        bert_df (list<str>): clinical trials data with inclusion and exclusion criteria split
        _nlp (None, spacy model): spacy model for embedding, ignored if custom embeddings are passed 
        stopwords (list): list of stopwords to remove
        criteria_col (str): criteria column in criteria_df
        class_col (str): column containing groups to plot top topics for
        nr_topics (str, int): number of topics for BERTopic to select, use 'auto' for DBSCAN auto selection
        
    """
    # set seed for reproducibility
    if seed is not None: umap_model = UMAP(random_state=seed)
        
    # get docs and embeddings
    docs = bert_df[criteria_col]
    embeddings = np.asarray([_nlp(c).vector for c in docs])
    
    # Train our topic model using our pre-trained sentence-transformers embeddings
    model = BERTopic(nr_topics=nr_topics, umap_model=umap_model).fit(docs, embeddings)
    bert_df["Topics"], bert_df["TopicProbs"] = model.transform(docs, embeddings, )

    return model, bert_df 

def updateBERTopicPerClassValues(model, bert_df, original_df, count_col="NCTId", 
                                 groupby="conditionMeshMainBranch", topic_values="Exclusion"):
    """
    Update topics per class from BERTopic models with correct (not exploded) group values 
    
    Params:
        model (BERTopic): BERTopic model fit on bert_df ["%sCriteriaEmbedClean"%topic_values] values
        bert_df (pd.DataFrame): dataframe containing exploded Eligibility Criteria values
        original_df (pd.DataFrame): dataframe containing non-exploded eligibility values
        count_col (str): for counting purposes
        groupby (str): column containing groups to count values for
        topic_values (str): Inclusion or Exclusion
        
    Returns:
        pd.DataFrame: topics_per_class from model.topics_per_class updated with non-exploded values
    
    """
    
    # get topics_per_class values (really only used to generate the dataframe columns)
    topics_per_class = model.topics_per_class(bert_df["%sCriteriaEmbedClean"%topic_values], 
                           topics=bert_df["Topics"], classes=bert_df[groupby])
    
    # update with unique values (Calculate (# trials in each group with topic / total trials in each group) 
    # not (# elibility lines in each group mapped to topic / total elibility lines in each group))
    topics_per_class_custom = original_df.explode("%sTopicNames"%topic_values).groupby(["%sTopicNames"%topic_values, groupby]).count()[[count_col]].reset_index()
    topics_per_class_custom["Topic"] = [int(t.split("_")[0]) for t in topics_per_class_custom["%sTopicNames"%topic_values]]

    # merge to topics_per_class
    topics_per_class_custom = topics_per_class_custom.set_index(["Topic", groupby])
    topics_per_class = topics_per_class.set_index(["Topic", "Class"])
    topics_per_class["Frequency"] = topics_per_class.index.map(topics_per_class_custom[count_col])

    # add totals
    topics_per_class = topics_per_class.reset_index()
    number_trials_per_mesh = dict(original_df.groupby([groupby]).count()[count_col])
    topics_per_class["Total"] = topics_per_class["Class"].map(number_trials_per_mesh)
    
    # add percentage
    topics_per_class["Percent"] = topics_per_class["Frequency"] / topics_per_class["Total"]

    return topics_per_class

### Outcomes cleaning
def cleanTimeValues(time_values):
    """
    Clean ClinicalTrial.gov Outcome Timeframe values
    
    Params:
        time_values (pd.Series): clinical_df["PrimaryOutcomeTimeFrame"] values
    
    Returns:
        list: cleaned time_values
    """
    # basic preprocessing
    time_values = ["---".join(p) for p in time_values]
    time_values = cleanTextValues(time_values, regex='[^\sA-Za-z0-9-, ><≤≥\.()]+')
    
    # basic time values
    for v in ["week", "day", "month", "year", "hour", "minute", "second"]:
        time_values = [p.replace(v+"s", v) for p in time_values]
        
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    for v in range(10):
        time_values = [p.replace(numbers[v], str(v)) for p in time_values]
    
    # clinical trial baseline values
    trial_times = {"baseline":"day 0", "immediately prior":"day 0", 
                   "before intervention":"day 0", "immediately before":"day 0"}
    for v in trial_times.keys():
        time_values = [p.replace(v, trial_times[v]) for p in time_values]
    
    return [p.split("---") for p in time_values]

def extractTimeframeValues(time_values):
    """
    Regex based time frame extraction for ClinicalTrial.gov data based on most common patterns
    Recommended that you run cleanTimeValues first for clean timeframe text data
    
    Params:
        time_values (pd.Series): PrimaryOutcomeTimeFrame CT.gov values

    Returns:
        list: extracted values (standardized to months) for each CT entry
    
    """
    # params
    extracted_time = []
    s = "day|month|year|hour|minute|week|second"
    
    # extract all time values from outcome timeframe
    for trial_time in time_values:
        trial_values = []
        #((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|(nov|dec)(?:ember)?)\D?(\d{1,2}\D?)?\D?((19[7-9]\d|20\d{2})|\d{2}
        #p = re.compile('(?P<timeframe>%s) \d+ to (?P<value>\d+) (?!%s)'%(s,s)) #"day XX to XX" or "XX and XX "
        
        for curr_time in trial_time:
            curr_values = []
            
            #"X-month" or "X month"
            p = re.compile('(?P<value>\d+)[\s\-]*(?P<timeframe>%s)'%s) 
            for c in p.findall(curr_time):
                curr_values.extend([(c[1], c[0])])
                
            #"(X) month"
            p = re.compile('(?P<value>\(\d+\))[\s\-]*(?P<timeframe>%s)'%s) 
            for c in p.findall(curr_time):
                curr_values.extend([(c[1], c[0].strip('\(\)'))])
                
            #"X-X minutes"
            p = re.compile('(?P<value1>\d+)\-+(?P<value2>\d+)[\s\-]*(?P<timeframe>%s)'%s) 
            for c in p.findall(curr_time):
                curr_values.extend([(c[2], c[0])])
                curr_values.extend([(c[2], c[1])])
            
            #"year X" 
            p = re.compile('(?P<timeframe>%s)[\s\-](?P<value>\d+)'%s) 
            curr_values.extend(p.findall(curr_time))
            
            #"day XX, XX, XX" or "day XX, XX, and XX" or "day XX, XX and XX"
            p = re.compile('(?P<timeframe>%s)\s(?P<value>\d+(?:\,\s\d+)+,?\s?(?:and)?\s?\d*(?:%s)?)'%(s,s))
            for list_time in p.findall(curr_time):
                if not any(d  in list_time[1] for d in "day|month|year|hour|minute|week|second".split("|")):
                    c = list_time[0]
                    list_time = list_time[1].replace(", and", ",")
                    list_time = list_time.replace("and", ",")
                    
                    for l in list_time.split(","):
                        l = l.strip()
                        curr_values.extend([(c, l)])
            
            trial_values.append(list(set(curr_values)))
            
        extracted_time.append(trial_values)
    return extracted_time

def convertTimeframe(value, start="day", end="month", round_dec=6):
    """
    Converts values from one timeframe to another
    Supported timeframes are "day", "month", "year", "minute", "second", "hour", "week"
    
    Params:
        value (int, float): value to convert
        start (str): original timeframe
        end (str): timeframe to convert to 
        round_dec (int, None): number of decimal spaces to round to
    
    Return:
        float: converted time value
    
    """
    timeframes = {"day":1, "month":float(1/30), "year":float(1/365), 
                  "minute":24*60, "second":24*60*60, "hour":24, "week":float(1/7)}

    if round_dec is not None: 
        return round(value * (float(timeframes[end])/float(timeframes[start])), round_dec)
    else:
        return (value * (float(timeframes[end])/float(timeframes[start])))

def convertOutcomeTimeframeValues(time_values, convert_to="day"):
    """
    Converts clinical trial outcome timeframe values
    Timeframe values should be extracted using extractTimeframeValues
    
    Params:
        time_values (pd.Series): CT.gov time values to convert
        convert_to (str): Supported formats are "day", "month", "year", "minute", "second", "hour", "week"
        
    Returns:
        list: converted time values
    """
    converted = []
    
    for times in time_values:
        curr_times = []
        
        for curr_t in times:
            curr_t = [convertTimeframe(value=float(t[1]), start=t[0], end=convert_to) for t in curr_t]
            curr_times.append(curr_t)
        
        converted.append(curr_times)
        
    return converted

def extractLongestTimeframePerOutcome(outcome_time):
    """
    Extracts longest timeframe for *each outcome* in a clinical trial
    Assumes standardized timeframes
    
    Params:
        outcome_time (list): CT.gov timeframe values for a single outcome
        
    Returns:
        float, float: longest time value for each clinical trial
    """
    
    max_time = max(outcome_time)
    return max_time, outcome_time.index(max_time)

def extractOverallLongestTimeframe(ct_timeframes):
    """
    Extracts overall longest timeframe in a clinical trial
    Assumes standardized timeframes
    
    Params:
        outcome_time (list): CT.gov timeframe values for a single outcome
        
    Returns:
        list, list: longest time value for each outcome and overall for clinical trial

    """
    overall = []
    per_outcome = []
    
    for curr_timeframe in ct_timeframes:
        # get max for each outcome
        outcome_longest = [max(t)  if len(t)>0 else np.nan for t in curr_timeframe]
        per_outcome.append(outcome_longest) 
        
        # get overall longest
        overall.append(max(outcome_longest))

    return per_outcome, overall

######################################################
### General data cleaning - non NLP related cleaning

def convertToDatetime(dates):
    """
    Format datetime to %Y-%m-%d from ClinicalTrials.gov 
    
    Params:
        dates (list<str>): Datetime values
        
    """
    dates = [s if type(s)!=str 
                    else np.nan if s ==""
                     else datetime.strptime(s, '%Y-%m-%d') if "-" in s
                     else datetime.strptime(s, '%B %d, %Y') if "," in s
                     else datetime.strptime(formatDate(s), '%m/%d/%y') if "/" in s
                     else datetime.strptime(s, '%B %Y') for s in dates]

    return [d if type(d)!=str else d.strftime('%Y-%m-%d') for d in dates]
  
def extractDictValuesFromList(dict_list, key):
    """
    Given a list of dictionary values, extract out values for a specific key
    
    Params: 
        dict_list (list<dict>): list of dictionaries to extract values from
        key_value (str): dictionary key to extract values of
    
    Returns:
        list<values> from specified key 
        
    Example: 
        multi_list = [{"A":1, "B":4}, {"A":2, "B":5}, {"A":3, "B":6}]
        extractDictValuesFromList(multi_list, key_value="A")
        >> [1,2,3]

    """

    return [c if type(c)==float else [x[key] if key in x.keys() else "NA" for x in list(c)] if c is not None else c for c in dict_list]

def formatDate(d, values='d/m/y', sep="/", ret_values='d/m/y'):
    """
    Formats date values to be zero-padded and converted to expected format

    Params:
        d (str): datetime formatted string
        values (str): format of d
        sep (str): separator for d
        ret_values (str): format to convert d into
        
    Returns:
        Zero-padded date string
    """
    d = d.split(sep)
    d = ["0"+curr_d if len(curr_d)==1 else curr_d for curr_d in d]
    values_dict = dict(zip(values.split(sep), d))
    for k in values_dict.keys():
        ret_values = ret_values.replace(k, values_dict[k])
    
    return ret_values

def cleanTextValues(text_field, regex='[^\sA-Za-z0-9- ><≤≥\.]+', 
                    stopwords=[], concat=None):
    """
    Basic text cleaning (stripping special characters, stopword removal, etc) from clinical trials field
    
    Params:
        elig_criteria (list, pd.Series): eligibility criteria from clinical trials data
        regex (str): regex values to capture
        stopwords (list): list of stopwords
        concat (str, None): concats document using string provided
        
    Returns:
        list: Cleaned values
    """
    
    # lowercase & remove trailing values from the inclusion/exclusion splitting
    text_field = [(str(n)).lower() for n in text_field]
    
    # strip special characters
    text_field = [i.replace("\n :", " ") for i in text_field]
    text_field = [i.replace("\n", " ") for i in text_field]
    text_field = [re.sub(regex, ' ', i) for i in text_field]      
    
    # remove stop words
    text_field = [n.split(" ") for n in text_field]
    text_field = [" ".join([w for w in t if not w.lower() in stopwords]) for t in text_field]
    
    # concatecate into documents
    #if concat is not None: text_field = "\n\n".join([n.replace("\n\n", "\n") for n in text_field])

    return text_field

def filterMultiIndex(counts_df, filter_cols=['MeshBranch', "EnrollmentType"], min_cat=2):
    """
    Filters dataframe based on existance of multiindex
    Eg. for default values, removes MeshBranch categories that do not have 2+ EnrollmentTypes
    
    """    
    # drop any categories without both actual and anticipated values
    counts_df["count"] = 1
    mesh_cat = counts_df.groupby(filter_cols).first().groupby(filter_cols[0]).count()
    mesh_cat = mesh_cat[mesh_cat["count"]>=min_cat]
    counts_df = counts_df[counts_df[filter_cols[0]].isin(mesh_cat.index)]
    
    return counts_df

def getTrialDurations(start_dates, completion_dates):
    """
    Calculate overall trial duration and time to primary completion 
    
    Params:
        startDates (list, pd.Series): 
        completionDates (list, pd.Series): 
        
    Returns:
        Trial duration (years, list)
    """ 
    durations = [c-s for c,s in zip(convertToDatetime(completion_dates), convertToDatetime(start_dates))]
    return [float("{:.3f}".format((a.days/365))) for a in durations]

#@st.experimental_singleton
def cleanDataset(clinical_df):
    """
    Basic cleaning 
    """
    # Add collaborators metadata
    clinical_df["CollaboratorName"] = extractDictValuesFromList(clinical_df["Collaborator"], key="CollaboratorName")
    clinical_df["CollaboratorClass"] = extractDictValuesFromList(clinical_df["Collaborator"], key="CollaboratorClass")
    clinical_df["NumberCollaborators"] = [len(x) if type(x)==list else 0 for x in clinical_df["CollaboratorName"]]
    clinical_df["PrimaryCollaboratorClass"] = collapseCollaboratorType(clinical_df["CollaboratorClass"], multiple="group")

    # Add trial duration metadata
    clinical_df["StartDate"] = convertToDatetime(clinical_df["StartDate"])
    clinical_df["PrimaryCompletionDate"] = convertToDatetime(clinical_df["PrimaryCompletionDate"])
    clinical_df["CompletionDate"] = convertToDatetime(clinical_df["CompletionDate"])

    clinical_df["StartYear"] = [s.year if type(s)!=str else np.nan for s in clinical_df["StartDate"]]
    clinical_df["PrimaryCompletionYear"] = [s.year if type(s)!=str else np.nan for s in clinical_df["PrimaryCompletionDate"]]
    clinical_df["CompletionYear"] = [s.year if type(s)!=str else np.nan for s in clinical_df["CompletionDate"]] 
    clinical_df["TrialDurationYears"] = getTrialDurations(clinical_df["StartDate"], clinical_df["CompletionDate"])
    clinical_df["TrialPrimaryDurationYears"] = getTrialDurations(clinical_df["StartDate"], clinical_df["PrimaryCompletionDate"])

    # Add location metadata for US and canada
    clinical_df["NumberLocations"] = [len(x) if type(x)==list else 0 for x in clinical_df["Location"]]

    for l in ["LocationFacility", "LocationStatus", "LocationCity", "LocationState", "LocationZip", "LocationCountry"]:
        clinical_df[l] = extractDictValuesFromList(clinical_df["Location"], key=l)

    getTrialLocations(clinical_df['LocationZip'], country="us")
    getTrialLocations(clinical_df['LocationZip'], country="ca")

    if "state_code" not in clinical_df:
        loc_df = mapTrialLocations(clinical_df[["NCTId", "LocationZip"]])
        clinical_df = clinical_df.merge(loc_df, how="left", left_on="NCTId", right_on="NCTId")

    # Average enrollment values
    clinical_df["EnrollmentCount"] = pd.to_numeric(clinical_df["EnrollmentCount"])
    clinical_df["EnrollmentAvg"] = clinical_df["EnrollmentCount"].divide(clinical_df["NumberLocations"], fill_value=0)

    all_stopwords = st.session_state.stopwords
    # Clean eligibility criteria
    if "InclusionCriteria" not in clinical_df.columns:
        elig_df = extractInclusionExclusionCriteria(clinical_df[["NCTId", "EligibilityCriteria"]])
        clinical_df = clinical_df.merge(elig_df, how="outer", left_on="NCTId", right_on="NCTId")

        clinical_df["IncCriteriaClean"] = cleanTextValues(clinical_df["InclusionCriteria"], stopwords=all_stopwords) 
        clinical_df["ExCriteriaClean"] = cleanTextValues(clinical_df["ExclusionCriteria"], stopwords=all_stopwords) 

    # Deprecated: Extract eligibility entities by Stanza (takes about 5 min to run for 350 trials)
    #clinical_df["InclusionEnts"], clinical_df["InclusionEntTypes"] = extractStanzaEnts(clinical_df["IncCriteriaClean"], stanza_nlp=stanza_nlp)
    #clinical_df["ExclusionEnts"], clinical_df["ExclusionEntTypes"] = extractStanzaEnts(clinical_df["ExCriteriaClean"], stanza_nlp=stanza_nlp)

    # Extract primary outcomes timeline, takes a couple of seconds
    clinical_df["PhaseClean"] = [np.nan if x is None else np.nan if type(x)==float else ", ".join(list(x)) for x in clinical_df["Phase"]]
    for k in ["Measure", "Description", "TimeFrame"]:
        clinical_df["PrimaryOutcome%s"%k] = extractDictValuesFromList(clinical_df["PrimaryOutcome"], key="PrimaryOutcome%s"%k)
        
    clinical_df["PrimaryTimeframeClean"] = cleanTimeValues(clinical_df["PrimaryOutcomeTimeFrame"])
    clinical_df["PrimaryTimeframeExtracted"] = extractTimeframeValues(clinical_df["PrimaryTimeframeClean"])
    clinical_df["PrimaryTimeframeExtractedMonths"] = convertOutcomeTimeframeValues(clinical_df["PrimaryTimeframeExtracted"], convert_to="month")
    clinical_df["PerPrimaryOutcomeLongestMonths"], clinical_df["OverallPrimaryOutcomeLongestMonths"] = extractOverallLongestTimeframe(clinical_df["PrimaryTimeframeExtractedMonths"])

    # Extract reference values
    for r in ["ReferenceCitation", "ReferencePMID", "ReferenceType"]:
        clinical_df[r] = extractDictValuesFromList(clinical_df["Reference"], key=r)

    
    # Extract conditions into MeSH branches using Scispacy
    if "conditionMeshMainBranch" not in clinical_df.columns:
        _nlp = st.session_state.nlp
        mesh_dict = st.session_state.mesh_dict
        clinical_df["conditionMesh"] = extractConditionMeshBranch(clinical_df["Condition"], **{"_nlp":_nlp, "mesh_dict":mesh_dict}) # 
        clinical_df["conditionMeshMainBranch"] = [collapseMeshEnts(m) for m in clinical_df["conditionMesh"]]

        # some manual mapping
        clinical_df["conditionMeshMainBranch"] = [m if m!="Unknown"
                                 else "Pathological Conditions, Signs and Symptoms" if "pain" in "".join(c).lower() 
                                else "Pathological Conditions, Signs and Symptoms" if "headache" in "".join(c).lower()
                                 else "Infections" if "hiv" in "".join(c).lower() 
                                 else "Infections" if "covid" in "".join(c).lower() 
                                 else "Infections" if "sars-cov" in "".join(c).lower() 
                                 else "Chemically-Induced Disorders" if "opioid" in "".join(c).lower() 
                                 else "Mental Disorders" if "eating disorder" in "".join(c).lower() 
                                 else "Nervous System Diseases" if "stroke" in "".join(c).lower() 
                                 else "Cardiovascular Diseases" if "atrial" in "".join(c).lower() 
                                 else "Nervous System Diseases" if "".join(c).lower() == 'als'
                                 else "Unknown" for m,c in zip(clinical_df["conditionMeshMainBranch"], clinical_df["Condition"]) ]

    
    return clinical_df


######################################################
### Loading data
def loadLocalNCTData(NCTId_list):
    """
    Loads in full set of data for each clinical trial 
    
    Params:
        NCTId_list (list<str>): list of NCT IDs to retrieve records for
    
    Returns:
        pd.DataFrame of full NCT records
    """
    
    # get all JSON files
    NCT_list = []
    
    for n in NCTId_list:
        # if clinical trial available, get JSON
        fpath = "./dataInput/AllAPIJSON/"+n[:-4]+"xxxx/"+n+".json"
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                j = json.load(f)
                NCT_list.append(j["FullStudy"]["Study"])
        
        # If not available, pull full record from ClinicalTrials.gov and download
        else:
            # pull record
            j = getFullRecordFromAPI(n, fmt="json")
            NCT_list.append(j.json()["FullStudiesResponse"]["FullStudies"][0]["Study"])
            
            # TODO: download to appropriate folder
            #_downloadClinicalTrialFromAPI(NCT)
            print("Retrieved record: "+n)
    
    clinical_df = pd.json_normalize(NCT_list, sep="-")
    clinical_df.columns = [n.split("-")[-1] for n in clinical_df.columns]

    return clinical_df

def queryClinicalTrialsAPI(query, return_fields, verbose=False, n_lim=None, 
                           study_url = "https://ClinicalTrials.gov/api/query/study_fields?", search_field=None):
    """
    Retrieve data from ClinicalTrials.gov
    Search ClinicalTrials.gov data using requests
    
    Params:
        query (str): text query
        return_fields (list<str>): fields to return, see 
        n_lim (int, None): limit to number of queries to return
        study_url (str): Clinical trials api link
        search_field (list<str>): limit ClinicalTrials.gov fields to search for query in
        // verbose (bool): print query numbers
        
    """

    # search for clinical trials and get data
    if len(query)==0: 
        return

    if len(return_fields) > 20:
        return("Too many search results to return! Max 20")

    
    # first search
    clinical_df = pd.DataFrame()
    curr_min = 1
    curr_max = 1000

    # this is a little confusing but
    # "fields"=which fields to return results for ="return_fields"
    # "field"=field to search for query in = "search_field"
    params = {"expr":query,
                "fmt":"JSON", 
                "fields" : ",".join(return_fields),
                "min_rnk":curr_min,
                "max_rnk":curr_max
            }
    if search_field is not None: params["field"] = search_field

    r = requests.get(study_url, params)
    #if len(r.text) < 1000: st.write(r.text) # QA 
    info = r.json()

    # get total number of studies
    if info["StudyFieldsResponse"]["NStudiesReturned"]>0:
        n_total = info["StudyFieldsResponse"]["NStudiesFound"]
        info = info['StudyFieldsResponse']["StudyFields"]
        clinical_df = pd.DataFrame(info)
    else:
        
        return "No studies found"
    
    if n_lim is not None: 
        n_total = min(n_total, n_lim)

    if n_total > curr_max:
        while n_total > curr_max:
            curr_min = curr_min + 999
            curr_max = min(n_total, curr_max + 999)

            # get new data
            params = {"expr":query,
                "fmt":"JSON", 
                "fields" : ",".join(return_fields),
                "min_rnk":curr_min,
                "max_rnk":curr_max
            }
            if search_field is not None: params["field"] = search_field

            r = requests.get(study_url, params)
            info = r.json()

            # append to dataframe
            if info["StudyFieldsResponse"]["NStudiesReturned"]>0:
                info = info['StudyFieldsResponse']["StudyFields"]
                curr_df = pd.DataFrame(info)
                clinical_df = clinical_df.append(curr_df)

            #if verbose: st.write("%d/%d records retrieved"%(curr_max,n_total))
    #else:
        #if verbose:  st.write("%d records retrieved"%n_total)

    # clean up values
    del clinical_df["Rank"]
    clinical_df = clinical_df.replace(np.nan, "")
    clinical_df = clinical_df.replace("\t", ' ')
    clinical_df = clinical_df.replace("\n", ' ')
    clinical_df = clinical_df.applymap(lambda x: "\t".join(x) if type(x)==list else x)
    clinical_df = clinical_df.replace(np.nan, "")

    # drop duplicates
    #clinical_df = clinical_df.set_index("NCTId")
    clinical_df = clinical_df.drop_duplicates()

    return clinical_df

def filterCTQueriesByCol(data_df, search_fields, queries):
    """
    Filters for clinical trials with exact match searches of queries in specified search_fields
    
    Params:
        data_df (pd.DataFrame): CT.gov data
        search_fields (list<str>): fields to search in
        queries (list<str>): queries to look for
    """
    NCT_list = []
    for s in search_fields:
        curr_search = data_df[["NCTId",s]]
        curr_search[s] = [n.lower() if type(n)==str else n for n in curr_search[s]]

        for q in queries:
            query_search = curr_search[curr_search[s].str.contains(q.lower(), na=False)]
            NCT_list.extend(query_search["NCTId"])
            
    return NCT_list

def getFullRecordFromAPI(NCTId, fmt="json"):
    """
    Get full clinical trials record using ClinicalTrials.gov API
    
    Params:
        NCTId (str): clinical trial ID to retrieve
        fmt (str): json or xml
        
    Returns:
        ClinicalTrials.gov response to retrieve full trial for given NCTId
    """
    return requests.get("https://www.clinicaltrials.gov/api/query/full_studies?expr=%s&fmt=%s"%(NCTId, fmt))

#@st.experimental_singleton
def getUserQuery(queries, fields_list, search_fields):
    """
    Retrieves full dataset for user query and saves to session state
    Cached to prevent repeat querying/random updates to session state

    TODO: allow for cached results to be loaded
    """
    # get clinical trials that match specific search fields
    # TODO: switch to https://www.clinicaltrials.gov/api/query/field_values?fmt=JSON&expr=query&field=NCTId
    full_query_df = pd.DataFrame(columns=["NCTId"])

    for q in queries:
        if len(q.strip())>0:
            q = '"'+q.strip()+'"'
            print(q)
            curr_df = queryClinicalTrialsAPI(q, fields_list, search_field=search_fields)
            curr_df["query"]=q
            full_query_df = full_query_df.append(curr_df)

    # filter
    full_query_df = full_query_df.drop_duplicates(subset=["NCTId"])
    NCT_list = filterCTQueriesByCol(full_query_df, search_fields, queries)
    full_query_df = full_query_df[full_query_df["NCTId"].isin(NCT_list)]

    # load all trials
    full_query_df = loadLocalNCTData(full_query_df["NCTId"])
    return full_query_df

    # TODO: also save a copy to disk 

def getSpacyNLP(model="en_core_sci_lg", linker="mesh"):
    """
    Load (sci)spacy pipeline
    """
    # TODO: figure out how to cache the scispacy linker so it doesn't keep pulling from the website
    #en_core_sci_sm #en_ner_bc5cdr_md (contains disease/chemical entities) #python -m spacy download en_core_web_md
    nlp = spacy.load(model, disable=['textcat', 'parser'])  
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": linker, "max_entities_per_mention":1})
    nlp.max_length = 1000000
    
    return nlp

def meshIDToBranchDict(get_dict=True):
    """
    Loads or generates (from XML file, link below) dataframe with ~60k MeSH IDs, MeSH term names, and associated branches
    Retrieved March 28, 2022: https://www.nlm.nih.gov/databases/download/mesh.html
    
    Params: 
        get_dict (bool): if False returns Mesh terms as Pandas DataFrame
    """
    
    if os.path.isfile("./dataOutput/meshDict.csv"):
        mesh_df = pd.read_csv("./dataOutput/meshDict.csv", index_col="id")
        if get_dict: return mesh_df.to_dict(orient="index")
        return mesh_df
    else:
        meshdict = {}

        curr_id = None
        curr_term = None
        main_branch = None
        curr_trees = []
        heading = False
        
        with open("./dataInput/desc2022.xml", "r") as fp:
            line = fp.readline()
            while line:
                if ("<DescriptorUI>" in line) and (curr_id is None):
                    curr_id = line.strip().replace("<DescriptorUI>", "").replace("</DescriptorUI>", "")

                elif ("<DescriptorName>" in line) and (curr_term is None):
                    curr_term = fp.readline()
                    curr_term = curr_term.strip().replace("<String>", "").replace("</String>", "")

                elif "<TreeNumberList>" in line:
                    treeline = fp.readline()
                    if "." in treeline: main_branch = treeline.strip().split(".")[0][-3:]
                    else: 
                        main_branch = treeline.strip().replace("<TreeNumber>", "").replace("</TreeNumber>", "")
                        heading = True

                    while "<TreeNumber>" in treeline:
                        treeline = treeline.strip().replace("<TreeNumber>", "").replace("</TreeNumber>", "")
                        curr_trees.append(treeline)
                        treeline = fp.readline()
                    
                elif "</DescriptorRecord>" in line:
                    meshdict[curr_id] = {"id":curr_id, "name":curr_term, "branches":curr_trees, "main_branch":main_branch, "heading":heading}
                    curr_trees = []
                    curr_id = None
                    curr_term = None
                    main_branch = None
                    heading = False

                line = fp.readline()

        # map values to headings
        mesh_df = pd.DataFrame.from_dict(meshdict, orient="index")
        metrics = pd.read_csv("./dataOutput/meshDiseaseHeadings.txt", sep=";", header=None)
        mesh_df = mesh_df.merge(metrics, left_on="main_branch", right_on=1, how="left")
        mesh_df = mesh_df.set_index("id", drop=True)
        mesh_df.columns = ["name", "branches", "main_branch", "heading", "branch_name", "main_branch_delete"]
        del mesh_df["main_branch_delete"]

        mesh_df.to_csv("./dataOutput/meshDict.csv")
        if get_dict: return mesh_df.to_dict()
        return mesh_df

######################################################
### Plotting functions

def plotCompletionYear(plots_df, colors, x_values="CompletionYear",xlabel="Completion Year", figsize=(8,5)):
    """
    Histplot of completion years

    Params:
        plots_df (pd.DataFrame): 
        colors (palette):
        x_values (str): 
        xlabel (str): 
        figsize (Tuple<int, int>): 

    """
    fig,ax = plt.subplots(figsize=figsize) 
    ax = sns.histplot(data=plots_df, x=x_values, binwidth=1, hue="StudyType", multiple="stack",  palette=colors)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=2022, ymin=ymin, ymax=ymax, ls='--', lw=2, color="gray")
    ax.set_xlabel(xlabel)
    #if len(ax.containers)>1: ax.bar_label(ax.containers[1], color="steelblue")
    #ax.bar_label(ax.containers[0], color="orange")
    
    return fig

def plotStudyDuration(plots_df, colors, values="StartToCompletionYears", order=["Observational", "Interventional"], figsize=(6, 6), ylabel="Trial duration (years)"):
    """
    TODO: DOCUMENTATION
    """
    plots_df["StartToCompletionYears"] = [(c-s).days/365 if type(s)!=str 
                                              else np.nan for s,c in zip(plots_df["StartDate"], plots_df["CompletionDate"])]
    #plots_df["StartToPrimaryCompletionYears"] = [(c-s).days/365 if type(s)!=str 
    #                                      else np.nan for s,c in zip(plots_df["StartDate"], plots_df["PrimaryCompletionDate"])]

    # plot distribution of study duration
    fig,ax = plt.subplots(figsize=figsize) 

    pointplot_df = plots_df[["StartToCompletionYears"]] #, "StartToPrimaryCompletionYears"
    pointplot_df = pointplot_df.unstack().reset_index()
    pointplot_df.columns = ["DurationType", "NCTIndex", "Years"]
    pointplot_df["StudyType"] = pointplot_df["NCTIndex"].map(dict(zip(plots_df.index, plots_df["StudyType"])))

    # DONE: change this to simple boxplot (easier to understand)
    ax = sns.boxplot(data=pointplot_df, x="StudyType", y="Years", palette=colors, order=order)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    return fig


def plotMeshBranchCounts(plots_df, mesh_col="conditionMeshMainBranch", min_count=5, hue=None, figsize=(8,15)):
    """
    TODO: documentation
    """
    # get counts for main mesh groups & drop groups with less than 3 trials
    counts_df = plots_df.copy(deep=True)
    counts_df["count"] = 1
    
    groupby = [mesh_col]
    if hue is not None: groupby.append(hue)
    counts_df = counts_df.groupby(groupby).sum()[["count"]].reset_index()
    counts_df = counts_df[counts_df["count"] >= min_count]
    counts_df = counts_df.sort_values("count", ascending=False)
    
    # plot values
    #st.write("Only main MeshBranch considered.")
    fig,ax = plt.subplots(figsize=figsize) 
    sns.barplot(data=counts_df, x="count", y=mesh_col, hue=hue)
    ax.set_ylabel('')
    ax.set_xlabel('Clinical trials')

    return fig


def plotCompareMeshGroupValues(mesh_df, n_col=3, hue=None, log_value=None,
                                      min_len=5, counts_col="EnrollmentCount", x_col="EnrollmentType",
                                      mesh_col="conditionMeshMainBranch", **kwargs):
    """
    # does actual and anticipated enrollment differ between DTx in different categories?
    # TODO: 
    """
    # get log values   
    if log_value is not None: 
        mesh_df[counts_col + " (log%s)"%(log_value)] = np.log(mesh_df[counts_col]) / np.log(log_value)
        counts_col = counts_col + " (log%s)"%(log_value)

    # drop any categories with fewer than expected number of studies
    mesh_df = mesh_df.groupby([mesh_col, x_col]).filter(lambda x : len(x)>=min_len)

    # filter to only Mesh categories with both actual and anticipated studies and sort values
    mesh_df = filterMultiIndex(mesh_df, filter_cols=[mesh_col, x_col], min_cat=2)
    mesh_df = mesh_df.sort_values(counts_col, ascending=False)

    # plotting
    g = sns.catplot(data=mesh_df, y=counts_col, x=x_col, col_wrap=n_col, hue=hue,
                    col=mesh_col, kind="box", height=7, aspect=0.8, **kwargs)
    for ax in g.fig.axes:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel("Enrollment")

    plt.subplots_adjust(wspace = 0.1)
    g.set_titles(template='{col_name}')

    return g, mesh_df

def plotGeographicDistributionbyState(locations_df, average="mean", values="EnrollmentAvg", title="Number of clinical trial locations by state", 
                                      log_value=None, scale_label="Count", return_df=False, figsize=(600,450)):
    """
    Plot geographic distribution of trial lovations by state_code, obtained using mapTrialLocations()
    Also requires a values column, if no value column is provided, defaults to 1 per trial location

    Params:
        locations_df (pd.DataFrame): contains state_code and values columns to plot
        average (str): type of average to take, "mean", "count"
        values (str, None): value for each trial location
        log_value (int, None): 
        figsize (tuple<int,int>): figure size in height x width in pixels
        TODO: documentation
    """
    # explode to get all trials
    locations_df = locations_df.explode("state_code")
    
    # default if no values provided
    if values is None: 
        locations_df["count"] = 1
        values="count"

    # get values
    plots_df = locations_df.groupby("state_code")[[values]].agg([average]).reset_index()

    if scale_label is None: scale_label = values+" "+average
    plots_df.columns = ["state_code", scale_label]
    
    # log if necessary
    if log_value is not None:
        old_scale_label = scale_label
        scale_label = old_scale_label + " (log%s)"%log_value
        plots_df[scale_label] = np.log(plots_df[old_scale_label]) / np.log(log_value)
    
    # plot
    fig = px.choropleth(
        locations=plots_df["state_code"], 
        color = plots_df[scale_label].astype(float), 
        locationmode="USA-states", 
        scope="usa", 
        color_continuous_scale = 'ylorrd', 
        labels={'color': scale_label},
        height=figsize[1],
        width=figsize[0], 
        )
    fig.update_layout(title_text=title, font_size=14)
    
    return fig

def plotPerPopulationTrial(values, state_code_dict, normalize=10e4, figsize=(600,450),title="Clinical trial locations per 100k population",
                           state_pop_file="./dataInput/censusPopulation.csv", label="count/100k"):
    """
    https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
    
    Params:
        values (dict): state 
        state_pop_file (str): state population file from census data
        normalize (int, None): value to normalize to, default per 100k people
    
    Returns: 
        px.choropleth
    """
    
    # read file
    state_df = pd.read_csv(state_pop_file)
    if normalize is not None: state_df["July2021Estimate"] = state_df["July2021Estimate"].divide(10e4)
    
    # format
    state_df["clinicalTrials"] = state_df["State"].map(values)
    state_df["clinicalTrialsPer%s"%normalize] = state_df["July2021Estimate"] / state_df["clinicalTrials"]
    state_df["state_code"] = state_df["State"].map(state_code_dict)

     # plot
    fig = px.choropleth(
        locations=state_df["state_code"], 
        color = state_df["clinicalTrialsPer%s"%normalize].astype(float), 
        locationmode="USA-states", 
        scope="usa", 
        color_continuous_scale = 'ylorrd', 
        labels={'color': label},
        height=figsize[1],
        width=figsize[0], 
        )
    fig.update_layout(title_text=title, font_size=14)

    return fig, state_df

def plotSponsorCollaborations(plots_df, sponsor_col="LeadSponsorClass", collab_col="PrimaryCollaboratorClass", 
                              explode_collaborators=False, height=650, width=475):
    """
    Plots Sankey diagram of primary sponsors and collaborator types
    Recommended that collaborator types are collapsed, or use explode_collaborators to see all collaborations
    
    Params:
        plots_df (pd.DataFrame): contains "SponsorType", "CollaboratorType" columns
        explode_collaborators (bool): if multiple collaborators are present, plots all possible sponsor-collaborator combinations for each trial
        height, width (int, int): plotly figure size
        
    Returns: 
        None
    
    """
   
    
    ### SPONSOR & COLLABORATOR TYPES
    # explode collabortors if multiple
    if explode_collaborators:
        plots_df[collab_col] = [["NONE"] if c is None else c for c in plots_df[collab_col]]
        plots_df = plots_df.explode(collab_col)
    
    # get counts
    plots_df["count"] = 1
    plots_df = plots_df.groupby(["LeadSponsorClass", "PrimaryCollaboratorClass"])[["count"]].count()
    plots_df = plots_df.reset_index()
    plots_df.columns = ["from_type", "to_type", "count"]

    # start of solution, define source and target of sankey from column concat
    plots_df = plots_df.assign(source=lambda d: d["from_type"],
              target=lambda d: d["to_type"] + "-Collab")
    
    # create color list
    colors_dict = {"INDUSTRY":"rgba(129, 77, 212, 1)", "NIH":"rgba(25, 115, 188, 1)",  #
                   "FED":"rgba(0, 124, 119, 1)", "OTHER_GOV":"rgba(140, 110, 75, 1)", 
                    "OTHER":"rgba(226, 111, 40, 1)",  "INDIV":"rgba(23, 166, 55, 1)", 
                   "UNKNOWN":"rgba(117, 64, 67, 1)", "NETWORK":"rgba(227, 90, 174, 1)",   
                   "MULTIPLE":"rgba(237, 235, 215, 1)", "NONE":"rgba(185, 80, 96, 1)"} 
    
    collab_dict = {}
    for k in colors_dict.keys():
        collab_dict[k+"-Collab"] = colors_dict[k]
    colors_dict.update(collab_dict)
    
    # get labels with manual ordering
    labels = list(set(plots_df["source"].append(plots_df["target"])))
    for n in ["OTHER_GOV", "NIH-Collab", "FED-Collab", "NETWORK-Collab", "OTHER_GOV-Collab", "MULTIPLE-Collab","OTHER", "UNKNOWN","NONE", "OTHER-Collab", "NONE-Collab","UNKNOWN-Collab",]:
        if n in labels:
            labels.append(labels.pop(labels.index(n)))
    
    indices = dict(zip(labels, range(len(labels))))
    
    # color values
    colors = [colors_dict[l] for l in labels]
    plots_df["node_color"] = plots_df["source"].map(colors_dict)
    plots_df["link_color"] = [c.replace("1)", "0.2)") for c in plots_df["node_color"]]

    # y values for sponsors
    sponsors_dict = [l for l in labels if "Collab" not in l] # 
    sponsor_count = plots_df.groupby("source").sum().loc[sponsors_dict]["count"]
    sponsor_count = [(i/sum(sponsor_count)) for i in sponsor_count]
    
    new_list = []
    curr_sum = 0
    for t in sponsor_count:
        new_list.append(t/2+curr_sum)
        curr_sum = curr_sum+t
    
    sponsors_dict = dict(zip(sponsors_dict, new_list))
    
    # y values for collab
    collab_dict = [l for l in labels if "Collab" in l] # 
    collab_count = plots_df.groupby("target").sum().loc[collab_dict]["count"]
    collab_count = [(i/sum(collab_count)) for i in collab_count]
    
    new_list = []
    curr_sum = 0
    for t in collab_count:
        new_list.append(t/2+curr_sum)
        curr_sum = curr_sum+t
        
    collab_dict = dict(zip(collab_dict, new_list))
    
    # location values
    y_location = [collab_dict[l] if "Collab" in l else sponsors_dict[l] for l in labels] #
    #y_location = [0.01 if i==0 else 0.99 if i==1 else i for i in y_location]

    # plot figures
    fig = go.Figure(data=[ 
        go.Sankey(
            arrangement = "snap",
            node = dict(pad = 10, 
                    label = labels,
                    x = [0.9 if "Collab" in l else 0.1 for l in labels],
                    y = y_location,
                    color = colors
            ),
                
            link = dict(source = plots_df["source"].map(indices), # indices correspond to labels, eg NIH, INDUSTRY, etc
                    target = plots_df["target"].map(indices),
                    value = plots_df["count"],
                    color = plots_df["link_color"]
            ),
        )], layout = go.Layout(autosize=False,width=width, height=height,
                               margin=go.layout.Margin(l=50,r=50, b=50, t=50, pad=5)),
    )

    fig.update_layout(title_text="Primary Sponsors & Collaborator Types",  font_size=14, )
    
    # update names
    new_labels = [t.split("-")[0] if "-" in t else t for t in fig.data[0]["node"]["label"]]
    for trace in fig.data:
        trace.update(node={"label":new_labels}, visible=True)

    return fig


def plotBERTopicsbyGroup(model, topics_per_class, top_n = 6, palette="viridis", 
                         figsize=(900,850), normalize=False, update_values=False,
                        topics=None, topic_names=None):
    """
    Plot top topic groups by mesh branch
    
    Params:
        model (BERTopic model): model fit on text data
        topics_per_class (pd.DataFrame): topics_per_class from model, recommend running updateBERTopicPerClassValues() if grouping values need to be updated
        top_n (int): number of top topics to show for each group, ignored if topics list specified
        palette (str): matplotlib palette
        figsize (tuple<int, int>): width, height
        update_values (bool): whether to update figure with topics_per_class values
        topics (list, None): topics to show
        topic_names (dict): {index:new_name} dictionary if you want to name your own topic values
    
    Returns:
        go.Figure
    """
    
    # Plot values per mesh class
    fig = model.visualize_topics_per_class(topics_per_class, top_n_topics=top_n, topics =topics,
                                           normalize_frequency=normalize, width=figsize[0], height=figsize[1])
    topics = [f["name"] for f in fig.data]
    color_dict = dict(zip(topics, sns.color_palette(palette, n_colors=len(topics)).as_hex()))

    # loop through each trace to update colors and values (if specified)
    for trace in fig.data:
        trace.update(marker={"color":color_dict[trace["name"]]}, visible=True)
        if update_values:
            # get values for topic
            curr_values = topics_per_class[topics_per_class["Name"] == trace["name"]].set_index("Class")
            curr_values = curr_values.loc[trace["y"]]
            
            # update values
            if normalize: trace.update(x=curr_values["Percent"])
            else: trace.update(x=curr_values["Frequency"])

    # update layout
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    fig.update_yaxes(title='', visible=True, showticklabels=True)
    fig.update_layout(title="", xaxis_title="Frequency")
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=1))
    fig.update_layout(xaxis = dict(tick0 = 0, dtick = 0.2))
    
    # update names
    if topic_names is not None: 
        fig.for_each_trace(lambda t: t.update(name = topic_names[t.name.split("_")[0]]))
    else:
        fig.for_each_trace(lambda t: t.update(name = ", ".join(t.name.split("_")[1:]).capitalize()))

    return fig


            
            
            
            