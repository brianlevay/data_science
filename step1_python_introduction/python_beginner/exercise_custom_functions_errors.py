## 3. Optional arguments ##

# Default code
def tokenize(text_string, special_characters, clean=False):
    if (clean == True):
        cleaned_story = clean_text(text_string, special_characters)
        story_tokens = cleaned_story.split(" ")
        return(story_tokens)
    else:
        uncleaned_tokens = text_string.split(" ")
        return(uncleaned_tokens)

tokenized_story = tokenize(story_string, clean_chars, True)
tokenized_vocabulary = tokenize(vocabulary, clean_chars)
misspelled_words = []

for word in tokenized_story:
    if word not in tokenized_vocabulary:
        misspelled_words.append(word)

## 5. Practice: creating a more compact spell checker ##

def clean_text(text_string, special_characters):
    cleaned_string = text_string
    for string in special_characters:
        cleaned_string = cleaned_string.replace(string, "")
    cleaned_string = cleaned_string.lower()
    return(cleaned_string)

def tokenize(text_string, special_characters, clean=False):
    cleaned_text = text_string
    if clean:
        cleaned_text = clean_text(text_string, special_characters)
    tokens = cleaned_text.split(" ")
    return(tokens)

final_misspelled_words = []

def spell_check(vocabulary_file, text_file, special_characters=[",",".","'",";","\n"]):
    misspelled_words = []
    file_vocab = open(vocabulary_file, "r")
    file_text = open(text_file, "r")
    raw_vocab = file_vocab.read()
    raw_text = file_text.read()
    file_vocab.close()
    file_text.close()
    tokenized_vocabulary = tokenize(raw_vocab, special_characters)
    tokenized_text = tokenize(raw_text, special_characters, True)
    for word in tokenized_text:
        if word not in tokenized_vocabulary and word != '':
            misspelled_words.append(word)
    return misspelled_words
    
final_misspelled_words = spell_check("dictionary.txt", "story.txt")
print(final_misspelled_words)
    

## 7. Syntax errors ##

def spell_check(vocabulary_file, text_file, special_characters=[",",".","'",";","\n"]):
    misspelled_words = []
    vocabulary = open(vocabulary_file).read()
    text = open(text_file).read()
    
    tokenized_vocabulary = tokenize(vocabulary, special_characters)
    tokenized_text = tokenize(text, special_characters, True)
    
    for ts in tokenized_text:
        if ts not in tokenized_vocabulary and ts != '':
            misspelled_words.append(ts)
    return(misspelled_words)

final_misspelled_words = spell_check(vocabulary_file="dictionary.txt", text_file="story.txt")
print(final_misspelled_words)

## 11. Traceback ##

def spell_check(vocabulary_file, text_file, special_characters=[",",".","'",";","\n"]):
    misspelled_words = []
    vocabulary = open(vocabulary_file).read()
    # Add ending parentheses.
    text = open(text_file).read()
    
    # Fix indentation.
    tokenized_vocabulary = tokenize(vocabulary, special_characters)
    tokenized_text = tokenize(text, special_characters, True)
    
    for ts in tokenized_text:
        if ts not in tokenized_vocabulary and ts != '':
            misspelled_words.append(ts)
    return(misspelled_words)

final_misspelled_words = spell_check(vocabulary_file="dictionary.txt", text_file="story.txt")
print(final_misspelled_words)