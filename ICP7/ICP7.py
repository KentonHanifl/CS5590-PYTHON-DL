import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import trigrams

#for part 4
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

##Parts 1 and 2
##approach taken from
##https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text

def tag_visible(element):
    '''
    Returns true or false depending on if the element in the result set is in a non-text tag.
    '''
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    '''
    Returns the space separated text elements in an html file
    '''
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)
        

cont = True
while cont:
    
    outfile = open('input.txt','w',encoding='utf8')

    site = input("Input a site.")

    try:
        file = requests.get(site)
        data = text_from_html(file.text)
        outfile.write(data)
           
    except Exception as e:
        print("There was a problem getting the site")
        print(e)
        
    finally:
        outfile.close()
        print("done. Output saved to input.txt")

    choice = input("Enter another site? y/n")
    if choice.lower() != 'y':
        cont = False

#part 3
def tokenize(text):
    '''
    returns a list of all of the words in a long string
    removes any blank "words", punctuation, ect.
    '''
    splitText = text.split(' ')
    for idx,s in enumerate(splitText):
        splitText[idx] = s.strip(",").strip(";").strip("(").strip(")").strip(".").strip('"').strip("'")
    for s in range(splitText.count('')):
        splitText.remove('')
    return splitText

inp = open('input.txt','r',encoding='utf8').read()
sentences = inp.split('.')
print("Tokens")
tokens = tokenize(inp)
print(tokens[0:100])

print("\n\n\nPOS")
pos = nltk.pos_tag(tokens)
print(pos[0:100])


print("\n\n\nstemming")
pStemmer = PorterStemmer()
stems = []
for t in tokens:
    stems.append(pStemmer.stem(t))
print(stems[0:100])

print("\n\n\nlemmatizing")
lemmatizer = WordNetLemmatizer()
lemmas = []
for word,tag in pos:
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if not wntag:
             lemma = word
    else:
             lemma = lemmatizer.lemmatize(word, wntag)
    lemmas.append(lemma)
print(lemmas[0:100])

##>>> for word, tag in pos_tag(word_tokenize(sent)):
##...     wntag = tag[0].lower()
##...     wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
##...     if not wntag:
##...             lemma = word
##...     else:
##...             lemma = wnl.lemmatize(word, wntag)
##...     print lemma

print("\n\n\nTrigrams")
trigramsout = trigrams(tokens)
for idx,i in enumerate(trigramsout):
    print(i)
    if idx == 100:
        break
    
print("\n\n\nNER")
NER = []
fourth = False
half = False
threefourth = False
l = len(sentences)
for idx,s in enumerate(sentences):
    NER.append(ne_chunk(pos_tag(wordpunct_tokenize(s))))
    if not fourth and idx>l/4:
        fourth = True
        print("25%")
    if not half and idx>l/2:
        half = True
        print("50%")
    if not threefourth and idx>(l*3)/4:
        threefourth = True
        print("75%")

print(NER[0:10])

#part 4

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

#bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("bayes:",score)

#knn
clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("knn:",score)

#stopwords
tfidf_Vect = TfidfVectorizer(ngram_range=(1,2),stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("knn w/o stopwords:",score)
