
# coding: utf-8

# 

# # SECTION 1: GETTING THE API DATA
# 
# This section is a one-time-use block of code meant to download the relevant data from the Entrez PubMed database and write it to a file. All future references to the data will come from the file itself, to save network bandwidth.
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

# In[42]:

import math
import time
from Bio import Entrez
Entrez.email = "brianjlevay@gmail.com"


# ## Defining the Function
# 
# This section defines a generic function for getting PubMed records and writing the relevant data to a file. The function accepts search terms and a filename as arguments. The flow is as follows:
# 
# 1. Open a file for writing
# 2. Perform an initial pubmed search (eSearch), and store the results on the server (usehistory)
# 3. Iteratively perform API calls (eFetch) to get the results, 10000 at a time
# 4. Write the relevant parts of each result (number, abstract, and mesh terms) to a file
# 5. Close the file

# In[43]:

def get_api_data(terms, filename):
    f = open(filename + '.txt', 'w')
    
    handle = Entrez.esearch(db='pubmed', term=terms, usehistory='y')
    search = Entrez.read(handle)
    handle.close()

    count = int(search['Count'])
    query = search['QueryKey']
    web = search['WebEnv']
    print("{} records found through eSearch.".format(count))
    
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
            abstract = ''
            try:
                abstract = entry['MedlineCitation']['Article']['Abstract']['AbstractText'][0].encode('cp1252', 'replace').decode('cp1252')
            except:
                abstract = 'Not available or not able to be printed.'
            f.write(abstract + '\n')
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

# In[44]:

# Need to specify obesity as the major MeSH descriptor (MajorTopicYN="Y") [majr] vs [majr:noexp]

obesity_terms = 'obesity[majr] 2000:2012[pdat]'

# Only need to run this function once to get all of the relevant API data

#get_api_data(obesity_terms, 'obesity_pubmed')


# # SECTION 2: CATEGORIZING MESH TERMS
# 
# This section uses the MeSH definitions file (desc2015.xml) to create a list of terms that refer to "Disease or Syndrome". The list will then be used to filter the MeSH terms in the publication results.

# ## Setting Up the Environment
# 
# This section imports BeautifulSoup, which is used for XML parsing.

# In[45]:

from bs4 import BeautifulSoup


# ## Defining the XML Parsing Function
# 
# The source file for the MeSH terms is a large (~300 MB) xml file, and my preferred XML parser (BeautifulSoup) doesn't perform very well under such a load. So, for this exercise, I will initially split the XML file into chunks using basic string techniques, and then I will apply the parser to each fragment. This is slow, but the memory footprint is smaller and it runs without crashing. I know there are better XML libraries, but this is what I've got for now.
# 
# This function opens the definitions file, extracts only the descriptors that match a SemanticTypeName specified as an argument, and writes those terms to another file. It's important to note that this function only considers semantic types listed under the preferred concept!

# In[46]:

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

# In[48]:

# Only need to run this function once to get all of the relevant terms

#get_descriptors('Disease or Syndrome', 'mesh_disease_syndrome_terms')

