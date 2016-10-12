## 2. Extract line numbers ##

raw_hamlet = sc.textFile("hamlet.txt")
split_hamlet = raw_hamlet.map(lambda line: line.split('\t'))
split_hamlet.take(5)

def fix_id(line):
    new_line = line.copy()
    old_id = line[0]
    new_id = old_id.replace("hamlet@","")
    new_line[0] = new_id
    return new_line
    
hamlet_with_ids = split_hamlet.map(lambda line: fix_id(line))

## 3. Remove blank values ##

hamlet_with_ids.take(5)

more_than_id = hamlet_with_ids.filter(lambda line: len(line) > 1)
hamlet_text_only = more_than_id.map(lambda line: [item for item in line if item != ""])

## 4. Remove pipe characters ##

hamlet_text_only.take(10)

def clean_pipes(line):
    new_line = []
    for item in line:
        if item != "|":
            clean_item = item.replace("|","")
            new_line.append(clean_item)
    return new_line
            
clean_hamlet = hamlet_text_only.map(lambda line: clean_pipes(line))