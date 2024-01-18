"""
Created Feb 10, 2022
Updated: August 29, 2022

@author: BrendaM
Butte Lab, UCSF
Bakar Computational Health Sciences Institutes
https://prsinfo.clinicaltrials.gov/definitions.html
"""

from ClinicalTrialFunctions import *

# settings
import warnings
warnings.filterwarnings('ignore')

sns.set_style("white")
sns.set_context("talk")
st.set_page_config(layout="wide")

######################################################
### Streamlit plotting functions
def plotDatasetOverview():
    """
    Number of trials over time, average duration of trials, metrics
    """
    # create containers
    col1, col2, col3 = st.columns([4, 3, 1])

    # load data
    plots_df = st.session_state.plot_values

    # plot counts by year
    colors = {'Observational': "steelblue", 'Interventional': "orange"}
    with col1:
        if len(plots_df)>0:
            fig = plotCompletionYear(plots_df, colors=colors)
            st.pyplot(fig)

        else: 
            st.write("Too few studies to plot")

    # calculate and plot duration
    with col2:
        fig = plotStudyDuration(plots_df, colors, order=["Observational", "Interventional"])
        st.pyplot(fig)
        #(pointplot_df[pointplot_df["StudyType"]=="Interventional"]["Years"].mean())

    # write number of interventional vs observational studies
    counts = plots_df.value_counts("StudyType")
    with col3:
        if "Interventional" in counts.index: 
            st.metric("Interventional trials", counts["Interventional"] , delta=None, delta_color="normal")
        else:
            st.metric("Interventional trials", 0, delta=None, delta_color="normal")
        if "Observational" in counts.index: 
            st.metric("Observational trials", counts["Observational"] , delta=None, delta_color="normal")
        else:
            st.metric("Observational trials", 0, delta=None, delta_color="normal")

        st.metric("Avg duration", "%.2f"%plots_df["StartToCompletionYears"].mean(), delta=None, delta_color="normal")

def plotSponsorsAndLocations():
    # create containers
    col1, buffer, col2 = st.columns([4, 3, 11])

    # load data
    plots_df = st.session_state.plot_values

    ### SPONSOR & COLLABORATOR TYPES
    with col1:
        ## Figure 2B: Sponsor type
        fig = plotSponsorCollaborations(plots_df[["NCTId", "LeadSponsorClass", "PrimaryCollaboratorClass"]],
                                  collab_col="PrimaryCollaboratorClass", explode_collaborators=False,
                                  height=450, width=325, link_color_alpha=0.7)
        fig.update_layout(paper_bgcolor='white')
        st.plotly_chart(fig)

    with col2:
        normalize = st.checkbox("Normalize to state population")
        figsize=(550,400)
        
        if normalize:
            locations_df = plots_df.explode(["LocationState"])
            
            state_code_dict = {}
            for m, n in zip(plots_df["LocationState"], plots_df["state_code"]):
                if m is not None and n is not None and len(m) == len(n): state_code_dict.update(dict(zip(m,n)))
            
            fig, state_df = plotPerPopulationTrial(locations_df.value_counts("LocationState"), 
                                                    state_code_dict, figsize=figsize,
                                                    normalize=10e4)
            st.plotly_chart(fig)
        else:
            ## Figure 2A: Locations by state
            fig = plotGeographicDistributionbyState(plots_df[["NCTId", "state_code"]], 
                values=None, average="sum", log_value=None, figsize=figsize)
            
            st.plotly_chart(fig)

def plotMeshBranchValues():
    # create containers
    col1, col2 = st.columns([3, 4])

    # load data
    plots_df = st.session_state.plot_values

    # remove unknown values
    plots_df = plots_df[plots_df["conditionMeshMainBranch"] != "Unknown"]
    
    ## Figure 3A: Mesh branch counts (at least 10 in group)
    with col1: 
        min_count = st.number_input("Min trial count", min_value=1, value=10)
        try:
            ax = plotMeshBranchCounts(plots_df[["conditionMeshMainBranch"]], min_count=min_count)
            st.pyplot(ax)
        except:
            st.write("Not enough values to plot")

    ## Figure 3B: Actual vs anticipated enrollment by MeSH branch
    with col2: 
        ## Statistics for actual vs anticipated values
        try:
            stats_df = mannWhitneyLongDf(plots_df, values_col="EnrollmentCount", min_values=min_count,
                      labels_col="EnrollmentType",subset_col="conditionMeshMainBranch")
            st.write("Actual vs Anticipated Enrolment")

            g, condition_df = plotCompareMeshGroupValues(plots_df[["EnrollmentCount", "conditionMeshMainBranch", "EnrollmentType"]], 
                n_col=3, counts_col="EnrollmentCount", x_col="EnrollmentType",  mesh_col="conditionMeshMainBranch", min_len=min_count)
            st.pyplot(g)

        except:
            st.write("Not enough values to plot")

def plotEligibilityCriteria():
    """
    Plot inclusion and exclusion criteria BERTopics
    """
    # create containers
    col1, col2 = st.columns([2, 8])

    # Plotting options
    with col1: 
        criteria = st.selectbox("Eligibility Criteria Type", options=["Inclusion", "Exclusion"])
        num_min = st.number_input("Min values per MeSH group", min_value=1, value=15)
        st.write("Click run to load topics")
        run_button = st.button("Run")

    clinical_df = st.session_state.plot_values
    all_stopwords = st.session_state.stopwords
    _nlp = st.session_state.nlp

    if run_button:
        st.write("Please be patient! This can take a minute to run")

        # Get values
        mesh_ind = clinical_df["conditionMeshMainBranch"].value_counts().loc[lambda x: x>num_min].index
        clinical_df = clinical_df[clinical_df["conditionMeshMainBranch"].isin(mesh_ind)]

        ## Figure 4A/B: Eligibility criteria topics by MeSH group
        topics = [0,1,2,3,4]

        # Merge to clinical_df
        bert_df = extractIndividualEligibility(clinical_df, criteria_col="%sCriteria"%criteria, stopwords=all_stopwords)
        model, bert_df = extractBERTopics(bert_df, _nlp, criteria_col="%sCriteriaEmbedClean"%criteria,
            seed=0, nr_topics='auto')

        clinical_df["%sTopics"%criteria] = clinical_df["NCTId"].map(dict(bert_df.groupby("NCTId")["Topics"].apply(set)))
        clinical_df["%sTopicNames"%criteria] = [[model.topic_names[i] for i in t] for t in clinical_df["%sTopics"%criteria]]

        topics_per_class = updateBERTopicPerClassValues(model, bert_df, clinical_df, topic_values="%s"%criteria)
        fig = plotBERTopicsbyGroup(model, topics_per_class, palette="viridis", update_values=True, 
                                   normalize=True, topics = topics)

        fig.update_layout(title="%s Criteria Topics"%criteria, xaxis_title="Frequency")
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=1))
        
        with col2: 
            st.plotly_chart(fig)
            
######################################################
### Loading dashboard
def _loadSessionStateFilterElements(plots_df):
    """
    Create sidebar elements for filtering
    Use session_state to save all values
    """
    st.sidebar.header("Study options")
    clinical_df = st.session_state.trials

    ### User options to filter data                                        
    # Types and status of studies to include
    default_status = ["Terminated", "Unknown status", "Withdrawn", "Suspended", "No longer available"]
    studyType = st.sidebar.multiselect("Study Type(s)", clinical_df["StudyType"].unique(), default=[p for p in clinical_df["StudyType"].unique() if p in ["Observational", "Interventional"]])
    studyStatus = st.sidebar.multiselect("Study Status", clinical_df["OverallStatus"].unique(), default=[p for p in clinical_df["OverallStatus"].unique() if p not in default_status])
    plots_df = plots_df[plots_df["StudyType"].isin(studyType)]
    plots_df = plots_df[plots_df["OverallStatus"].isin(studyStatus)]

    # Get only trials in certain time frame
    minStartYear = min([int(y) for y in clinical_df["StartYear"] if type(y)!=float])
    maxEndYear = max([int(y) for y in clinical_df["CompletionYear"] if type(y)!=float])
    startToEnd = st.sidebar.slider("Study Timeframe", minStartYear, maxEndYear,  (max(2010,minStartYear), min(2030, maxEndYear)))
    plots_df = plots_df[plots_df["StartDate"] > str(startToEnd[0]-1)+"-12-31"]
    plots_df = plots_df[plots_df["CompletionDate"] < str(startToEnd[1]+1)+"-01-01"]

    # FDA regulated devices
    fdaReg = st.sidebar.checkbox("Limit to FDA regulated devices", value=False)
    if fdaReg: plots_df = plots_df[plots_df["IsFDARegulatedDevice"] == "Yes"]

    # sponsor type
    sponsors = st.sidebar.multiselect("Sponsor Types", clinical_df["LeadSponsorClass"].unique(), default=[p for p in clinical_df["LeadSponsorClass"].unique()])
    plots_df = plots_df[plots_df["LeadSponsorClass"].isin(sponsors)]

    # reset button

    # update plot_values
    st.session_state.plot_values = plots_df

@st.cache_resource(show_spinner=False)
def loadGlobalObjects():
    """
    Load session state values

    """
    spacy_nlp = getSpacyNLP(model="en_core_sci_sm") #en_core_sci_lg "./dataInput/en_core_sci_sm-0.5.0")
    
    all_stopwords = spacy_nlp.Defaults.stop_words
    all_stopwords |= {"patient", "subject", "participant", "studies", "study", "individual", "e.g.",  "diagnosis", "participation", "participate"} 
    
    st.session_state.nlp = spacy_nlp
    st.session_state.stopwords = all_stopwords

    #mesh_dict = meshIDToBranchDict()
    #st.session_state.mesh_dict = mesh_dict


def loadUserQueryDashboard():
    """
    Loads dashboard elements and retrieves user query(s)
    """

    # TODO: allow for list of NCT codes for user input

    # load fields
    fields_list = ["NCTId", "OverallStatus", "StartDate", "EligibilityCriteria","CompletionDate",
                            "StudyType", "BriefTitle","DesignPrimaryPurpose", "LeadSponsorClass",  "CollaboratorClass", 
                            "DetailedDescription", "IsFDARegulatedDevice", "IsFDARegulatedDrug","InterventionType",
                            "BriefSummary", "OfficialTitle", "Keyword",  "InterventionDescription", "InterventionName",                       
                           ]

    # TODO: load search fields for user input
    search_fields = ["BriefSummary", "InterventionName", "InterventionDescription", "BriefTitle",
                     "Keyword", "DetailedDescription", "OfficialTitle", "EligibilityCriteria"]


    # get user input values or load default values
    search_type = st.sidebar.radio("Search", ["DTx Clinical Trials", "Upload custom file"], index=0)
    full_query_df =  pd.DataFrame(columns=["NCTId"])

    # get clinical trials that match specific search fields
    # TODO: switch to https://www.clinicaltrials.gov/api/query/field_values?fmt=JSON&expr=query&field=NCTId
    # load values from CT.gov or user input
    if search_type == "DTx Clinical Trials": #"Search ClinicalTrials.gov":
        # load values from CT.gov or user input
        #query = st.sidebar.text_area("Search terms (one query per line)", value="Digital therapeutic")
        #queries = [q.strip() for q in query.split("\n") if len(q)>1]
        #full_query_df = _getUserQuery(queries, fields_list, search_fields)
        
        full_query_df = pd.read_parquet("./exampleFile/DTxClinicalTrials.parquet.gzip")
        #st.write(full_query_df)

    else:
        uploaded_files = st.sidebar.file_uploader("Upload custom file", type=".gzip", accept_multiple_files=True)

        if len(uploaded_files) > 0:
            for file_name in uploaded_files: 
                curr_df = pd.read_parquet(file_name)
                full_query_df = full_query_df.append(curr_df)

    
    st.session_state.trials = full_query_df

    #if len(full_query_df) > 0:
        # clean and save to session state
        #full_query_df = cleanDataset(full_query_df) #, _nlp=spacy_nlp
        #st.session_state.trials = full_query_df

def plotDashboard():
    """
    Plotting values
    """
    # load values for plotting
    clinical_df = st.session_state.trials
    _loadSessionStateFilterElements(clinical_df)

    # Plot number and duration of interventional vs observational studies
    with st.expander("Dataset overview"): plotDatasetOverview()

    # Plot primary sponsor + collaborator class Sankey diagram & primary sponsor map
    with st.expander("Sponsor and collaborator metrics"): plotSponsorsAndLocations()
    
    # Plot conditions mapped to standard MESH values
    with st.expander("Standardized condition analysis"): plotMeshBranchValues()

    # Plot frequency of eligibility criteria topics across MeSH groups
    with st.expander("Eligibility criteria analysis"): plotEligibilityCriteria()

    # Supplemental figures
    with st.expander("Supplemental figures"):
        print("TODO")

    # allow download

    # allow reset

######################################################
### Main
loadGlobalObjects()    
loadUserQueryDashboard()
if "trials" in st.session_state: plotDashboard()

