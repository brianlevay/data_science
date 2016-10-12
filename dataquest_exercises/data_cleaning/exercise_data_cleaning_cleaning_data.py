## 4. Reading in the data ##

import pandas as pd

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

data = {}

for file in data_files:
    path = "schools/" + file
    file_data = pd.read_csv(path)
    
    parts = file.split('.')
    name = parts[0]
    data[name] = file_data

## 5. Exploring the SAT data ##

print(data['sat_results'].head(5))

## 6. Exploring the other data ##

for key in data:
    print(data[key].head(5))

## 7. Reading in the survey data ##

# read in the data and check the shape of each DataFrame

all_survey = pd.read_csv("schools/survey_all.txt",delimiter="\t",encoding="Windows-1252")
d75_survey = pd.read_csv("schools/survey_d75.txt",delimiter="\t",encoding="Windows-1252")
print(all_survey.shape)
print(d75_survey.shape)

# concatenate the surveys according to the instructions, and check the new shape

survey = pd.concat([all_survey, d75_survey],axis=0)
print(survey.shape)

# to verify the shape, generate the set of column names from both and look at length

all_survey_cols = list(all_survey.columns)
d75_survey_cols = list(d75_survey.columns)
unique_cols = set(all_survey_cols + d75_survey_cols)
print(len(unique_cols))

# print first 5 rows, as instructed

print(survey.head(5))

## 8. Cleaning up the surveys ##

cols = ["DBN", "rr_s", "rr_t", "rr_p", "N_s", "N_t", "N_p", "saf_p_11", "com_p_11", "eng_p_11", "aca_p_11", "saf_t_11", "com_t_11", "eng_t_11", "aca_t_11", "saf_s_11", "com_s_11", "eng_s_11", "aca_s_11", "saf_tot_11", "com_tot_11", "eng_tot_11", "aca_tot_11"]

survey = survey.assign(DBN = survey['dbn'])
survey = survey[cols]
data['survey'] = survey

print(data['survey'].shape)

## 9. Inserting DBN fields ##

data['hs_directory'] = data['hs_directory'].assign(DBN = data['hs_directory']['dbn'])

def left_pad(val):
    val_str = str(val)
    if len(val_str) == 1:
        return "0" + val_str
    else:
        return val_str

padded_csd = data['class_size']['CSD'].apply(left_pad)
new_dbn = padded_csd + data['class_size']['SCHOOL CODE']
data['class_size'] = data['class_size'].assign(DBN = new_dbn)

print(data['class_size'].head(5))

## 10. Combining the SAT scores ##

data['sat_results']['SAT Math Avg. Score'] = pd.to_numeric(data['sat_results']['SAT Math Avg. Score'], errors="coerce")
data['sat_results']['SAT Critical Reading Avg. Score'] = pd.to_numeric(data['sat_results']['SAT Critical Reading Avg. Score'], errors="coerce")
data['sat_results']['SAT Writing Avg. Score'] = pd.to_numeric(data['sat_results']['SAT Writing Avg. Score'], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results']['SAT Math Avg. Score'] + data['sat_results']['SAT Critical Reading Avg. Score'] + data['sat_results']['SAT Writing Avg. Score']

print(data['sat_results']['sat_score'].head(5))

## 11. Parsing coordinates for each school ##

import re

def get_latitude(string):
    found = re.findall("\(.+, .+\)", string)
    if len(found) > 0:
        parts = found[0].split(',')
        lat = parts[0][1:len(parts[0])]
        lon = parts[1][0:len(parts[1])-1]
        return lat
    else:
        return string

data['hs_directory']['lat'] = data['hs_directory']['Location 1'].apply(get_latitude)

print(data['hs_directory'].head(5))

## 12. Extracting the longitude ##

import re

def get_longitude(string):
    found = re.findall("\(.+, .+\)", string)
    if len(found) > 0:
        parts = found[0].split(',')
        lat = parts[0][1:len(parts[0])]
        lon = parts[1][0:len(parts[1])-1]
        return lon
    else:
        return string

data['hs_directory']['lon'] = data['hs_directory']['Location 1'].apply(get_longitude)

data['hs_directory']['lat'] = pd.to_numeric(data['hs_directory']['lat'], errors="coerce")
data['hs_directory']['lon'] = pd.to_numeric(data['hs_directory']['lon'], errors="coerce")

print(data['hs_directory'].head(5))