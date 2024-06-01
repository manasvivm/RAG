import nltk
nltk.download('all')
import re
import string
import gensim 
from gensim.parsing.preprocessing import remove_stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from nltk import pos_tag,word_tokenize,ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame

from nltk.corpus import wordnet,stopwords
import spacy
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from flask import Flask, render_template, request, jsonify


sample = open("Abstract.txt", "r") 
s = sample.read() 



app = Flask(__name__)


def stem_sentence(sentence):
  words=word_tokenize(sentence)
  #lemmatizer = WordNetLemmatizer()
  
  
  stemmer = SnowballStemmer("english")
  new_words=[]
  for i in words:
    new_words.append(stemmer.stem(i))
    new_words.append(" ")
  return "".join(new_words)  
  


def clean_sentence(sentence, stopwords=True):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    
    
    
    if stopwords:
         sentence = remove_stopwords(sentence)
    
   
    
    return sentence


def get_cleaned_sentences(sents,stopwords=True):    
    
    cleaned_sentences=[]

    for i in sents:
        
        cleaning=clean_sentence(i,stopwords)
        cleaned=stem_sentence(cleaning)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences



def create_document_term_matrix(sen,vectorizer):
  doc_term_matrix=vectorizer.fit_transform(sen)
  return DataFrame(doc_term_matrix.toarray(), columns=vectorizer.get_feature_names_out())
  



def calculate_cosine_similarity(df_list,sentences,question):
  a=[]
  for i in range(len(df_list)-1):
    sim=1 - spatial.distance.cosine(df_list[i], question)
    t=(sim,sentences[i])
    a.append(t)
  a.sort(reverse=True)
  n=[]
  for i in range(3):
    n.append(a[i][1])
    #print("*",n[i])
   
    
  return n  


def questiontype( question):
        questiontags = ['WP','WDT','WP$','WRB']
        question_POS = pos_tag(word_tokenize(question.lower()))
        
        question_Tags=[]
        for token in question_POS:
            if token[1] in questiontags:
              question_Tags.append(token)
                
                
        if len(question_Tags)==1 and question_Tags[0][0]!= 'what' :
          return True
        else:
          return False  
        

from nltk.util import ngrams
def n_gram_similarity(question,n):
  q=list(ngrams(word_tokenize(question.lower()),1))
  a=0
  b=0
  c=0
  t=[]
  for i in q:
    if i in list(ngrams(word_tokenize(n[0].lower()),1)):
      a=a+1
  for i in q:
    if i in list(ngrams(word_tokenize(n[1].lower()),1)):
      b=b+1
  for i in q:
    if i in list(ngrams(word_tokenize(n[2].lower()),1)):
      c=c+1        
  d=max(a,b,c)
  if a == d:
    t.append(n[0])
  if b == d:
    t.append(n[1]) 
  if c ==d:
    t.append(n[2])
  print()  
  #print("Selected Sentence:",t[0])  
  return t  





def answertype(question,df_list,sentences):
  nlp = spacy.load('en_core_web_sm')

  if (questiontype(question)):
    t='DESCRIPTIVE'
    flag=0
    word=word_tokenize(question.lower())
  
    if 'who' in word:
      t='PERSON'
    elif 'where' in word:
      t='GPE'
    elif 'how' in word and 'many' in word and  'age' in word or 'duration' in word or 'long' in word or 'days'in word or 'years' in word or'months' in word:
      t='DATE' 
    elif 'how' in word and 'many' in word :
       t = 'CARDINAL'  
    elif 'when' in word  or 'age' in word or 'period' in word or 'duration' in word  or 'old' in word or 'long' in word:
      t='DATE'
    elif 'how' in word  and 'long' in word or 'often' or 'age' in word or 'years' in word:
      t='DATE' 
    elif 'what' in word and 'time' in word or 'duration' in word or 'period' in  'word'  :
      t='DATE' 
    i=len(df_list)-1  
    n=calculate_cosine_similarity(df_list, sentences,df_list[i])
    n=n_gram_similarity(question,n)
    #print("Most relevant sentence", n[0])
    #print("ANSWER TYPE:",t)
    key = n[0]
    spdoc = nlp(key)
    entity_type=[]
    for ent in spdoc.ents:
       if ent.label_ == t:
          entity_type.append(ent.text)
    if len(entity_type) == 1:
      #print("ANSWER TYPE:", t)
       print("ANSWER:", entity_type[0])
       return entity_type[0]  
    if len(entity_type) == 0:
      #print("ANSWER TYPE:", t) 
       print(n[0])
       return  n[0]
    if len(entity_type) > 1:
      #print("Answer Type:",t)  
      key_question = question
      q=[]
      spdoc = nlp(key_question)
      for ent in spdoc:
        if ent.pos_ == 'NOUN' or ent.pos_ =='ADJ' :
          q.append(ent.text)
  
      key_answer = n[0]
      a = []
      spd = nlp(key)
      for ent in spd:
        if ent.pos_ == 'NOUN'or ent.pos_ =='ADJ' :
          a.append(ent.text)
  #s=[sentence.index(i) for i in t]
      s=[]
      w=[]
      for i in entity_type:
       s.append(n[0].index(i))
      for i in range(len(s)):
        w.append(0)

    
      for i in q:
        try:
           factor= n[0].index(i)
           for j in range(len(s)):
              w[j]=w[j]+(abs(s[j]-factor))
        except:
           continue    
      m=min(w)
      u=[]
      for i in range(len(s)):
        if w[i] == m:
           #print(entity_type[i])
           u.append(entity_type[i])
      print("ANSWER:",u[0])   
      return u[0]
      

  

    
  else:
    t='DESCRIPTIVE'
    #print("ANSWER TYPE:",t)
    i=len(df_list)-1  
    n=calculate_cosine_similarity(df_list, sentences,df_list[i])
    #n = n_gram_similarity(question, n)
    for j in n:
         print(j) 















@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    #print(input)
    question=[input]

    question=["What happens upon photolysis?"]
    for j in question:
    #j=TextBlob(j)
    #j=str(j.correct())
    
        qq=[]
        qp=[]
        que=sent_tokenize(j)
        
        qq.append(que)
        qp.append(j)
        questiontags = ['WP','WDT','WP$','WRB']
        question_POS = pos_tag(word_tokenize(j.lower()))
                
        question_Tags=[]
    for token in question_POS:
        if token[1] in questiontags:
            question_Tags.append(token)
    
    
    if len(question_Tags)>1:
        if ' and ' in j :
            pos=j.lower().find('and')
            qq=[]
            qp=[]
            qp.append(j[:pos])
            qp.append(j[pos+1:])
            qq.append(sent_tokenize(j[:pos])) 
            qq.append(sent_tokenize(j[pos+1:])) 
    

    for k in range(len(qp)):   
        


        sentences=sent_tokenize(s)
        #q contains a list of cleaned sentence tokens of question
        q=get_cleaned_sentences(qq[k],stopwords=True)
        #preprocessed contains a list of cleaned sentence tokens of the reference text
        preprocessed=get_cleaned_sentences(sentences,stopwords=True)
    
        preprocessed.append(q[0])
        i=len(preprocessed)-1
        
    
        tfidf_vect=TfidfVectorizer()
        df=create_document_term_matrix(preprocessed,tfidf_vect) 
        df_list = df.values.tolist()
        
    return answertype(qp[k],df_list,sentences)


def get_Chat_response(text):

    # Let's chat for 5 lines
    
        return  "HIIIIIII"


if __name__ == '__main__':
    app.run()
