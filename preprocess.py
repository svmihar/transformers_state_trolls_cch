import os
import string


if 'stopwords.txt' not in os.listdir():
    os.system('wget https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt -O stopwords.txt')

STOPWORDS = set([a.strip() for a in open('stopwords.txt').readlines()])

def remove_punctuation(s: str) -> str:
    temp = s
    for p in string.punctuation:
        temp.replace(p, '')
    return ' '.join([a for a in temp.split() if a])

def remove_stopwords(s: str)->str:
    cleaned = [word for word in s.split() if word.lower() not in STOPWORDS]

    return ' '.join(cleaned)


