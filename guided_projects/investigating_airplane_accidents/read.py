file = open("AviationData.txt","r")

## DATA STRUCTURES ##
## NOTES: This project does not want operations nested for efficiency
# First, read the data into a list of strings
aviation_data = []
for line in file:
    aviation_data.append(line)
        
# Then, read the data into a list of lists
aviation_lists = []
for line in aviation_data:
    cols = line.split(" | ")
    aviation_lists.append(cols)
    
# Then, later search it
lax_code = []
for row in aviation_lists:
    for col in row:
        if col == "LAX94LA336":
            lax_code.append(cols) 

# Now, read the data into a list of dictionaries
aviation_dict_list = []
names = aviation_data[0].split(" | ")
for line in aviation_data[1:]:
    cols = line.split(" | ")
    row_dict = {}
    for j,col in enumerate(cols):
        row_dict[names[j]] = col
    aviation_dict_list.append(row_dict)

#Then, later search it
lax_dict = []
for line in aviation_dict_list:
    for key in line:
        if line[key] == "LAX94LA336":
            lax_dict.append(line)
#print(lax_dict)


## USING THE STRUCTURES ##
# Accidents by US State #
states = ['AK','AL','AR','AZ','CA','CO','CT','DE','FL','GA',\
          'HI','IA','ID','IL','IN','KS','KY','LA','MA','MD',\
          'ME','MI','MN','MO','MT','MS','NC','ND','NE','NH',\
          'NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC',\
          'SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
state_accidents = {}
for line in aviation_dict_list:
    location = line["Location"]
    loc = location.split(", ")
    if len(loc) > 1:
        state = loc[1]
        if state in states:
            if state in state_accidents:
                state_accidents[state] += 1
            else:
                state_accidents[state] = 1

sorted_accidents = sorted(state_accidents,key=state_accidents.get,reverse=True)
highest = sorted_accidents[0]
number = state_accidents[highest]
print("{0} had the most accidents, with {1}".format(highest,number))

# Fatalities and injuries by month
mo_dict = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May',\
           '06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct',\
           '11':'Nov','12':'Dec'}
monthly_injury_dict = {}

for line in aviation_dict_list:
    date = line["Event Date"].split("/")
    fatalities = line["Total Fatal Injuries"]
    injuries = line["Total Serious Injuries"]
    try:
        tot = int(fatalities) + int(injuries)
    except:
        tot = 0
    try:
        mo = mo_dict[date[0]]
        if mo in monthly_injury_dict:
            monthly_injury_dict[mo] += tot
        else:
            monthly_injury_dict[mo] = tot
    except:
        pass

print(monthly_injury_dict)