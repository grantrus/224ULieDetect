def possessive_pronouns(sentence):
    #need to parse things like "theyre"
    l = ["i", "me", "my", "mine", "we", "us", "our", "ours"] 
    count = 0
    for w in sentence.split(' '):
        if w in l:
            count += 1
    return count

def thirdperson_pronouns(sentence):
    l = ["he", "him", "his", "her", "hers", "they", "them", "their", "theirs"]
    count = 0
    for w in sentence.split(' '):
        if w in l:
            count += 1
    return count

def negations(sentence):
    l = ["no", "not", "neither", "never", "no one", "nobody", "none", "nor", "nothing", "nowhere"]
    count = 0
    for w in sentence.split(' '):
        if w in l:
            count += 1
    return count

def cognitive_complexity(sentence):
    l = ["than", "rather than", "whether", "as much as", "whereas", "though", "although", "even though", "while", "if", "only if",
    "unless", "until", "providing that", "assuming that", "even if", "in case", "in case that", "lest"]
    count = 0
    for w in l:
        count += sentence.count(w)
    return count

def append_linguistic_general(df, f, name):
    counts = []
    for s in df.statement:
        counts.append(f(s))
    df[name] = counts

def apppend_linguistic(df):
    append_linguistic(df, possessive_pronouns, 'possessive')
    append_linguistic(df, negations, 'negations')
    append_linguistic(df, cognitive_complexity, 'complexity')