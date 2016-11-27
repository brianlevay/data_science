
# coding: utf-8

# # PROJECT SUMMARY: WHICH DISEASES ARE COMORBID WITH OBESITY?
# 
# ## Overview
# 
# This project was given to me as a data science exercise. The "who gave it to me" and "why" parts don't matter here, only the "what did they want" and the "how did they want me to do it". The goal of the project is to determine which diseases or syndromes are *comorbid* with obesity. In other words, the goal is to determine which diseases occur at higher rates in obese populations relative to non-obese populations. All of the data must come from PubMed/MEDLINE databases that are part of the National Center for Biotechnology Information (NCBI).
# 
# ## Methods
# 
# Some of the methods for this project are required, and others are left for me to decide. The initial, expected workflow is as follows:
# 
# 1. Retrieve the records for all papers published between 2000 and 2012 that have the term 'Obesity' as a major topic (note: 'Obesity' and 'Obesity, Morbid' are two different descriptors, but records containing either as major topics are returned using 'Obesity')
# 2. Using the list of all possible medical descriptors (MeSH terms), create a smaller list of terms that only refer to diseases or syndromes
# 3. Identify which papers talk about each disease in the new list, and extract any relevant data (number of references, odds ratio values, etc.)
# 4. Try to answer the question, if possible: which diseases are comorbid with obesity?
# 
# Documentation for the Entrez APIs (to access the PubMed data) can be found here:
# [Entrez Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
# 
# ## Desired Output
# 
# The ultimate goal of the project is to determine which diseases have a higher incidence rates in obese populations relative to non-obese populations (odds ratios > 1). The goal is *NOT* to determine which diseases are more severe in obese populations. You can see an illustration of the desired output in the next section.
# 
# It's important to note that this is a **Descriptive** study, and not a **Predictive** or **Prescriptive** study. The goal here is to understand what the existing data says, not to build machine learning models or decision trees (proper ones, for actually making decisions).
# 
# ## Data Limitation
# 
# One of the key assumptions that will be carried through the rest of this project is that odds ratios, statistical significances, etc. are *NOT* stored in the publication record metadata. The only place that data can be retrieved (using the approach requested) is via the abstracts. If I'm wrong, and the metadata *DOES* store that information, this project is a whole lot easier!

# # SECTION 0: EXAMPLE OF WHAT WE WANT
# 
# This section below generates a synthetic dataset using Monte Carlo techniques to highlight the type of visual (and statistical) product that we want. This is just meant to be a guide so that visually-inclined readers can see the "single figure summary" that we're after.

# ## Setting Up the Environment
# 
# This section imports the main math and plotting libraries that we'll use throughout the project

# In[138]:

import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Defining the Function
# 
# This section creates the function that generates a synthetic dataset for our example plot

# In[117]:

def create_synthetic_OR_plot():
    X_locations = [x for x in range(0,10)]
    X_labels = [chr(x+65) for x in range(0,10)]

    data = []
    means = {}

    for location in X_locations:
        p_disease_not_obesity = np.random.random_sample() * (0.5) + 0.02
        p_disease_obesity = np.random.random_sample() * (0.5) + 0.12
        papers = int(np.random.random_sample() * (100-5) + 1)
        disease_results = []
        for i in range(0,papers):
            observations = int(np.random.random_sample() * (1000 - 100) + 100)
            bias_disease_not_obesity = np.random.random_sample() * (0.2) - 0.1
            bias_disease_obesity = np.random.random_sample() * (0.2) - 0.1
            p_eff_disease_not_obesity = p_disease_not_obesity + bias_disease_not_obesity
            p_eff_disease_obesity = p_disease_obesity + bias_disease_obesity
            p_eff_disease_not_obesity = max(0.01,min(p_eff_disease_not_obesity,0.99))
            p_eff_disease_obesity = max(0.01,min(p_eff_disease_obesity,0.99))
            samples_disease_not_obesity = np.random.binomial(1, p_eff_disease_not_obesity, observations)
            samples_disease_obesity = np.random.binomial(1, p_eff_disease_obesity, observations)
            fraction_disease_not_obesity = sum(samples_disease_not_obesity) / observations
            fraction_disease_obesity = sum(samples_disease_obesity) / observations
            if fraction_disease_not_obesity != 0:
                odds_ratio = fraction_disease_obesity / fraction_disease_not_obesity
                disease_results.append(odds_ratio)
        data.append(disease_results)
        means[location] = np.mean(disease_results)

    sorted_locs = sorted(means, key=means.__getitem__, reverse=True)
    data_sorted = []
    labels_sorted = []
    for i in range(0,len(X_locations)):
        data_sorted.append(data[sorted_locs[i]])
        labels_sorted.append(X_labels[sorted_locs[i]])

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1,1,1)
    ax.boxplot(data_sorted)
    ax.plot([0,11],[1,1],color="black")
    ax.set_yscale('log')
    ax.set_xticklabels(labels_sorted)
    ax.set_xlabel("Disease or Syndrome")
    ax.set_ylabel("P(Disease|Obesity) / P(Disease|NotObesity)")
    ax.set_title("Synthetic Data Showing Diseases Ranked By Mean Odds Ratios")
    plt.show()


# ## Running the Function to Generate an Example Plot
# 
# Re-run this section below to see how the randomness in the function can create a range of variances!

# In[118]:

create_synthetic_OR_plot()


# # SECTION 1: GETTING THE API DATA
# 
# This section is a one-time-use block of code meant to download the relevant data from the Entrez PubMed database and write it to a file. All future references to the data will come from the file itself, to save network bandwidth. This step is critical, though. If the wrong search terms are used, or the incorrect data is extracted and stored, the entire rest of the analysis can be compromised! The search terms used will be discussed below.
# 
# **Storage Format for Each Publication:**
# 
# PUBMED_ID  
# Number  
# ABSTRACT  
# Abstract text  
# MESH  
# List of MeSH terms

# ## Setting Up the Environment
# 
# This section imports the necessary modules to implement the API calls. The main package used for this section is the Entrez submodule from Biopython.

# In[119]:

import time
from Bio import Entrez
Entrez.email = "brianjlevay@gmail.com"


# ## Defining the Function
# 
# This section defines a generic function for getting PubMed records and writing the relevant data to a file. The function accepts search terms, a filename, and a keyword to determine whether to fetch the results. An initial PubMed search is performed (eSearch), and the results are stored on the server by using the 'usehistory' keyword. The search key and web environment terms are returned, and if the user opts to fetch the results, the records are downloaded in batches, 10000 at a time (using eFetch). The records are stripped down and the relevant information is stored in a local file for later access.

# In[139]:

def get_api_data(terms, filename, want_fetch=False):
    handle = Entrez.esearch(db='pubmed', term=terms, usehistory='y')
    search = Entrez.read(handle)
    handle.close()

    count = int(search['Count'])
    query = search['QueryKey']
    web = search['WebEnv']
    print("{} records found through eSearch.".format(count))
    
    if want_fetch == True:
        f = open(filename + '.txt', 'w')
        
        max_ret = 10000
        steps_tot = math.ceil(count / max_ret)
        steps = [x*max_ret for x in range(0,steps_tot)]
        total_records = 0

        for step in steps:
            time.sleep(30)
            handle = Entrez.efetch(db='pubmed', query_key=query, WebEnv=web, retmode='xml', retstart=step, retmax=max_ret)
            fetch = Entrez.read(handle)
            handle.close()
            print("Step {}: API batch returned.".format(step))

            for entry in fetch:
                total_records += 1
                f.write('PUBMED_ID\n')
                f.write(entry['PubmedData']['ArticleIdList'][0] + '\n')
                f.write('ABSTRACT\n')
                try:
                    abstract = entry['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                    if abstract == '':
                        abstract = 'No abstract available'
                    try:
                        f.write(abstract + '\n')
                    except:
                        try:
                            abstract = abstract.encode('cp1252', 'replace').decode('cp1252')
                            f.write(abstract + '\n')
                        except:
                            f.write('Abstract could not be printed.\n')
                except:
                    f.write('No abstract available.\n')
                f.write('MESH\n')
                mesh_str = ''
                for mesh in entry['MedlineCitation']['MeshHeadingList']:
                    mesh_str += mesh['DescriptorName'] + '; '
                f.write(mesh_str[0:len(mesh_str)-2] + '\n')    
                f.write('\n')

        f.close()
        print("{} records retrieved via eFetch and written to the file.".format(total_records))


# ## Running the Function to Gather the Data
# 
# This block of code uses the generalized function defined above to retrieve the data from PubMed. This function only needs to be run once, and afterwards, the data will be stored in a local file. 
# 
# Getting the search terms right for this exercise is critical, but an individual could likely spend days trying to understand the nuances of the descriptors / subheadings / modifiers used to retrieve data from the database. Initially, I chose to use a simple, high-level search just using 'obesity' and a date range, but this **retrieved a lot of irrelevant records**. I tried a modified search using 'obesity' with a subheading of 'morbidity', and the results appear to be more topical. However, the choices made at this step will ripple through the rest of the project, so it's important to be aware of the tradeoffs.

# In[147]:

# Need to specify obesity as the major MeSH descriptor (MajorTopicYN="Y") [majr]

obesity_terms = 'obesity[majr] 2000:2012[pdat]'
obesity_morbidity_terms = 'obesity/morbidity[majr] 2000:2012[pdat]'

# Only need to run this function once to get all of the relevant API data

# get_api_data(obesity_morbidity_terms, 'obesity_pubmed', want_fetch=True)


# # SECTION 2: CATEGORIZING MESH TERMS
# 
# This section uses the MeSH definitions file (desc2015.xml) to create a list of terms that refer to "Disease or Syndrome". The list will then be used to filter the MeSH terms in the publication results.
# 
# The MeSH descriptors file is large, so it will not be stored along with this code and the intermediate data products. Instead, you can download it from the following ftp site. I did not directly link the file below, because this is one you don't want to accidently start downloading with a careless click!
# 
# Location for File:  
# ftp://nlmpubs.nlm.nih.gov/online/mesh/2015/

# ## Setting Up the Environment
# 
# This section imports BeautifulSoup, which is used for XML parsing.

# In[122]:

from bs4 import BeautifulSoup


# ## Defining the XML Parsing Function
# 
# The source file for the MeSH terms is a large (~300 MB) xml file, and my preferred XML parser (BeautifulSoup) doesn't perform very well under such a load. So, for this exercise, I will initially split the XML file into chunks using basic string techniques, and then I will apply the parser to each fragment. This is slow, but the memory footprint is smaller and it runs without crashing. I know there are better XML libraries, but this is what I've got for now.
# 
# This function opens the definitions file, extracts only the descriptors that match a SemanticTypeName specified as an argument, and writes those terms to another file. It's important to note that this function only considers semantic types listed under the preferred concept!

# In[123]:

def get_descriptors(semantic_type, filename):
    f = open('desc2015.xml', 'r')
    xml_contents = f.read()
    f.close()

    f = open(filename + '.txt', 'w')

    header_content = '<?xml version="1.0"?>\n' + '<!DOCTYPE DescriptorRecordSet SYSTEM "desc2015.dtd">\n' +         '<DescriptorRecordSet LanguageCode = "eng">\n' + '<DescriptorRecord DescriptorClass = "1">\n'
    xml_contents = xml_contents.replace(header_content, '')
    descriptors = xml_contents.split('</DescriptorRecord>\n<DescriptorRecord DescriptorClass = "1">')

    for descriptor in descriptors:
        descriptor = '<DescriptorRecord DescriptorClass = "1">\n' + descriptor + '</DescriptorRecord>'
        desc_soup = BeautifulSoup(descriptor, "xml")
        name = desc_soup.DescriptorName.String.get_text()
        semantic_tags = desc_soup.ConceptList.find('Concept', PreferredConceptYN='Y').find_all('SemanticTypeName')
        semantic_types = set()
        for tag in semantic_tags:
            semantic_types.add(tag.get_text())
        if semantic_type in semantic_types: 
            f.write(name + "\n")
    
    f.close()


# ## Running the Function to Output the Descriptors
# 
# This block of code runs the function defined above to generate a file with a list of applicable descriptors. You only need to run this once, and all future data access will come from the newly created file.

# In[124]:

# Only need to run this function once to get all of the relevant terms

# get_descriptors('Disease or Syndrome', 'mesh_disease_syndrome_terms')


# # SECTION 3: READING THE DATA FROM THE FILES
# 
# This section loads the data from the raw PubMed records (previously stored from the API calls) and the disease terms (previously stored from the descriptors list) into their respective data structures for use.

# In[152]:

f = open('mesh_disease_syndrome_terms.txt', 'r')
disease_terms = f.read()
f.close()

disease_terms = set(disease_terms.split('\n'))
print("{} terms in the disease list.".format(len(disease_terms)))

f = open('obesity_pubmed.txt', 'r')
pubmed_raw = f.read()
f.close()

pubmed_raw = pubmed_raw.split('\n\n')
pubmed_raw = pubmed_raw[0:len(pubmed_raw)-1]
print("{} records in the raw PubMed data.".format(len(pubmed_raw)))

pubmed_records = []
for record in pubmed_raw:
    record_dict = {}
    lines = record.split('\n')
    record_dict['ID'] = lines[1]
    record_dict['ABSTRACT'] = lines[3]
    record_dict['MESH'] = lines[5].split("; ")
    pubmed_records.append(record_dict)
    
print("\nEXAMPLE RECORD\n")
for item in pubmed_records[0]:
    print("{0}: {1}".format(item,pubmed_records[0][item]))


# # SECTION 4: COUNTING PAPERS THAT MENTION DISEASES
# 
# This section determines the number of papers that mention each disease as a MeSH keyword. It's important to note that the number of papers talking about a disease (in conjunction with obesity) actually tells us nothing about comorbidity, unless we make some strong assumptions about the *contents* of the papers. 
# 
# Papers that mention 'obesity' and disease 'A' together could be looking at:
# 
# 1. The prevalence or severity of disease 'A' in obese populations
# 2. The prevalence or severity of obesity in populations with disease 'A'
# 3. The prevalence or severity of some other issue ('Complication') in populations with both disease 'A' and obesity
# 
# Even in the subset of papers that study diseases in obese populations, there are multiple explanations for why two diseases have different citation counts:
# 
# 1. Disease 'A' is more common in obese populations than disease 'B', so 'A' gets more attention (what we want to know)
# 2. Disease 'A' is much more severe in obese populations relative to disease 'B', so 'A' gets more attention
# 3. The comorbidity or severity of disease 'A' in obese populations is harder to ascertain than disease 'B', and therefore more studies have been conducted to try to reduce the uncertainty
# 4. Disease 'A' may have more treatment options available and/or is considered easier to treat, so 'A' gets more attention
# 5. The research groups that study the relationships between disease 'A' and obesity might prefer breaking up their studies into smaller papers, or they might just publish more papers in general
# 
# There are almost certainly other reasons not outlined above. I'll talk about these issues more, in a bit.

# ## Counting the Occurrences
# 
# This section gets the number of occurrences of each disease term in MeSH keywords. It also gets the number of disease terms per paper. The algorithm works by iterating through the records, and for each record, it iterates through the MeSH term. If a term is found in the "disease_terms" list, then that term is incremented by 1 in disease_counts. In addition, each valid disease term in a record is counted, and the total number of valid terms per record is stored in the diseases_per_paper dictionary.

# In[142]:

disease_counts = {}
diseases_per_paper = {}

start_time = time.time()
for record in pubmed_records:
    term_ct = 0
    for term in record['MESH']:
        if term in disease_terms:
            if term in disease_counts:
                disease_counts[term] += 1
            else:
                disease_counts[term] = 1
            term_ct += 1
    if term_ct in diseases_per_paper:
        diseases_per_paper[term_ct] += 1
    else:
        diseases_per_paper[term_ct] = 1
end_time = time.time()

print("Time for algorithm: {} seconds\n".format(round(end_time - start_time),2))


# ## Looking at the Raw Data
# 
# This section makes some general observations about the results.

# In[143]:

import pandas as pd
disease_cts_df = pd.DataFrame.from_dict(disease_counts, orient='index')
disease_cts_df.reset_index(inplace=True)
disease_cts_df.columns = ['Disease','Cts']
disease_cts_df.sort_values('Cts', ascending=False, inplace=True)
disease_cts_df


# In[144]:

pct_zero = 100*diseases_per_paper[0]/len(pubmed_records)
pct_one = 100*diseases_per_paper[1]/len(pubmed_records)
pct_more = 100 - pct_one - pct_zero

more_than_one_df = disease_cts_df.loc[disease_cts_df['Cts'] > 1]
more_than_five_df = disease_cts_df.loc[disease_cts_df['Cts'] > 5]
more_than_ten_df = disease_cts_df.loc[disease_cts_df['Cts'] > 10]
more_than_twenty_df = disease_cts_df.loc[disease_cts_df['Cts'] > 20]
more_than_hundred_df = disease_cts_df.loc[disease_cts_df['Cts'] > 100]
more_than_twohund_df = disease_cts_df.loc[disease_cts_df['Cts'] > 200]

total_mentions = disease_cts_df['Cts'].sum()
papers_with_one_or_more = len(pubmed_raw) - diseases_per_paper[0]

print("Number of Disease Terms Per Paper")
print(diseases_per_paper, "\n")
print("Percentage of Papers With No Disease Terms in MeSH Keywords: {0:.2f} %".format(pct_zero))
print("Percentage of Papers With One Disease Term in MeSH Keywords: {0:.2f} %".format(pct_one))
print("Percentage of Papers With Two or More Disease Terms in MeSH Keywords: {0:.2f} %\n".format(pct_more))

print("Number of diseases or syndromes with >   0 mentions: {}".format(len(disease_cts_df)))
print("Number of diseases or syndromes with >   1 mentions: {}".format(len(more_than_one_df)))
print("Number of diseases or syndromes with >   5 mentions: {}".format(len(more_than_five_df)))
print("Number of diseases or syndromes with >  10 mentions: {}".format(len(more_than_ten_df)))
print("Number of diseases or syndromes with >  20 mentions: {}".format(len(more_than_twenty_df)))
print("Number of diseases or syndromes with > 100 mentions: {}".format(len(more_than_hundred_df)))
print("Number of diseases or syndromes with > 200 mentions: {}\n".format(len(more_than_twohund_df)))

print("Total number of times disease or syndrome terms were mentioned: {}".format(total_mentions))
print("Total number of papers with at least one disease or syndrome term: {}".format(papers_with_one_or_more))


# ## Plotting the Data

# In[153]:

df_to_plot = more_than_twenty_df
cutoff = 20

max_X = len(df_to_plot)
X_vals = [x for x in range(0,max_X)]

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1,1,1)
ax.bar(X_vals,df_to_plot['Cts'],align='center',color='lightblue')
ax.set_xticks(X_vals)
ax.set_xticklabels(df_to_plot['Disease'], rotation=90)
ax.set_xlim([-1,max_X])
ax.set_xlabel('Disease or Syndrome')
ax.set_ylabel('Number of Occurrences as MeSH Terms')
ax.set_title('Number of Times a Disease Was Mentioned in Association with Obesity (2000-2012) [> '+str(cutoff)+' Mentions, MeSH]')
plt.show()


# ## Discussion
# 
# First, it's worth noting that the majority of the PubMed records that we retrieved (~60 %) don't discuss any diseases or syndromes in association with obesity. Approximately 23 % of the records contain only one associated disease or syndrome keyword, and the rest contain two or more. Two records even contain 10 disease keywords in its MeSH terms!
# 
# Second, it's worth noting that, although there were 468 different diseases or syndromes mentioned at least once, only 123 were mentioned more than 5 times. You can see if the figure above (only showing terms mentioned > 20 times) that there is a long tail, with a large number of diseases only discussed (alongside obesity) in a relatively small number of papers.
# 
# So, which diseases are discussed most often alongside obesity? 
# 
# 1. Cardiovascular Diseases
# 2. Hypertension
# 3. Diabetes Mellitus, Type 2
# 4. Metabolic Syndrome X
# 5. Diabetes Mellitus
# 
# It's pretty clear that many of the disease / syndrome terms are related in nature. If I had more of a medical background (and understood the diseases better), I'd group similar conditions to better understand the classes of problems being studied.
# 
# So, what can we make of this data? As discussed at the beginning of this section, more studies (or more papers) do not necessarily mean higher incidence rates in obese populations! First, we have to consider that many papers talking about 'obesity' and 'disease A' aren't necessarily looking at the *relationship* between the two. They are looking at some third factor in populations with both 'obesity' and 'disease A'. Second, even if the studies are *about* the relationship between 'obesity' and 'disease A', we don't know whether the studies were looking at the odds of the disease in obese people or the odds of obesity in diseased people. Third, citation counts don't really tell us about odds ratios. It could be that the diseases with more attention are more prevalent (what we want), more severe, more treatable, more ambiguous (unclear odds ratios), etc. The only way to truly determine which of these diseases are comorbid (odds ratios > 1) is to extract the data from the records themselves.

# # SECTION 5: EXTRACTING DATA FROM ABSTRACTS
# 
# Finding the number of papers that mention a disease (in conjunction with obesity) only gets us so far towards answering our question. The pitfalls were discussed extensively in the previous section, so they won't be repeated here. What we really need to know is what the papers actually **say** about the disease in relation to obesity, and in particular, what we want to know is the **odds ratios** for the diseases in obese vs non-obese populations.
# 
# It's possible that a more refined PubMed search could eliminate many of the irrelevant records and return more epidemiology-oriented papers, making our job easier. As specified in the introduction, I'm carrying the assumption that the statistical results of the papers are not recorded in the metadata. If I am wrong (possible!), then the nature of this project changes quite a bit. However, for now, we're going to try to work with our very unconstrained dataset.
# 
# First, we'll take a closer look at some samples of the data, and then we'll try a few brute-force methods for gathering data. Text processing, in the best of scenarios (well-structured data), is challenging. In a scenario such as this (free-form text, no expectation of common language terms or writing styles), given the limited time, it may be impossible. If this was a more substantial project, the two areas that would need more focus would be: refining the initial search, and building out a decent set of text search algorithms.

# ## Looking at a Sample of Abstracts
# 
# In this section, we'll print a few records for the most commonly cited term. It's worth a good, qualitative look at the data to get a feel for how different authors might or might not summarize the relevant data in the abstracts.

# In[154]:

disease_sought = disease_cts_df['Disease'].iloc[0]
max_records = 5

print("A sample of records with the term '{}':\n".format(disease_sought))

returned_records = 0
for record in pubmed_records:
    for term in record['MESH']:
        if term == disease_sought:
            print('ABSTRACT')
            print(record['ABSTRACT'], "\n")
            print('MESH')
            print(record['MESH'], "\n")
            print('------------------------------------------------------\n')
            returned_records += 1
            break
    if returned_records == max_records:
        break


# ## Attempting to Get Values from the Text
# 
# In this section, we'll attempt to get some of the desired odds ratios out of the abstracts. The approach that we'll use here is a very simple one, using regex searchs with a few key expressions. We'll almost certainly miss reported odds ratios using this technique. In addition, we won't be able to distinguish P(Disease|Obesity) from P(Obesity|Disease), and we won't be able to know which disease corresponds to which odds ratio if multiple are reported in a study. Language processing is complicated, and in a real study, this section would require the bulk of the allocated time.

# In[216]:

import re

study_results = []
start_time = time.time()
for record in pubmed_records:
    study_diseases = []
    study_values = []
    has_or_terms = False
    for mesh_term in record['MESH']:
        if mesh_term in disease_terms:
            study_diseases.append(mesh_term)
    if len(study_diseases) > 0:
        text = record['ABSTRACT'].lower().replace('~','')
        search1 = re.findall(r'or = (?:\d*\.)?\d+', text)
        search2 = re.findall(r'o.r. = (?:\d*\.)?\d+', text)
        search3 = re.findall(r'odds ratio = (?:\d*\.)?\d+', text)
        search4 = re.findall(r'odds ratio is (?:\d*\.)?\d+', text)
        search5 = re.findall(r'odds ratio of (?:\d*\.)?\d+', text)
        if len(search1) > 0:
            study_values += search1
        if len(search2) > 0:
            study_values += search2
        if len(search3) > 0:
            study_values += search3
        if len(search4) > 0:
            study_values += search4
        if len(search5) > 0:
            study_values += search5
        if len(study_values) > 0:
            study_results.append([record['ID'],study_diseases, study_values])
end_time = time.time()

print("Total Number of Records Found with OR Values: {}\n".format(len(study_results)))
print("Examples of Results:\n")
for i in range(0,30):
    print(study_results[i][0], " ", study_results[i][1], " ", study_results[i][2])


# ## Validating the Results
# 
# In this section, we'll pull the abstracts for a few records and compare them to the results obtained above. If the odds ratios that we scraped appear to be in-context and match the type of data we're looking for, we might have some more confidence in our methods.

# In[ ]:




# ## Refining the Results
# 
# Content goes here...

# In[ ]:




# ## Plotting the Data
# 
# Content goes here...

# In[ ]:




# ## Discussion
# 
# Content goes here...

# # CONCLUSIONS
# 
# Content goes here...
