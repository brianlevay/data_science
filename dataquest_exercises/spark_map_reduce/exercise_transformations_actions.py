## 1. Introduction to the data ##

raw_hamlet = sc.textFile("hamlet.txt")
print(raw_hamlet.take(5))

## 2. Map ##

split_hamlet = raw_hamlet.map(lambda line: line.split("\t"))

## 4. FlatMap ##

def hamlet_speaks(line):
    id = line[0]
    speaketh = False
    
    if "HAMLET" in line:
        speaketh = True
    
    if speaketh:
        yield id,"hamlet speaketh!"

hamlet_spoken = split_hamlet.flatMap(lambda x: hamlet_speaks(x))
hamlet_spoken.take(10)

## 5. Filter using a named function ##

def filter_hamlet_speaks(line):
    if "HAMLET" in line:
        return True
    else:
        return False

hamlet_spoken_lines = split_hamlet.filter(lambda line: filter_hamlet_speaks(line))
hamlet_spoken_lines.take(5)

## 6. Actions ##

spoken_count = hamlet_spoken_lines.count()
spoken_101 = (hamlet_spoken_lines.collect())[100]