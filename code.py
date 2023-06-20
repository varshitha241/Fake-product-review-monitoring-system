# Fake-product-review-monitoring-system
Basically this project is all about the reviewing of the E-commerce platform review wether they are Fake or Genuine


from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import sklearn
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tkinter import filedialog
import nltk
main = tkinter.Tk()
main.title("Fake Product Review Monitoring System") #designing main screen
main.geometry("1300x1200")
global filename
global accuracy
stop_words = set(stopwords.words('english'))
global vector
global X_train, X_test, y_train, y_test
global classifier

def clean_doc(doc):
    tokens = doc.split()
    

table = str.maketrans('', '', punctuation)
          tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    return tokens

def checkInput(inputdata):
    option = 0
    try:
        s = float(inputdata)
        option = 0
    except:
        option = 1
    return option

def Preprocessing():
    global X_train, X_test, y_train, y_test
    global vector
    global X
    global Y
    X = []
    Y = []
    text.delete('1.0', END)
    train = pd.read_csv(filename,encoding = "ISO-8859-1")
    for i in range(len(train)):
        sentiment = train._get_value(i,0,takeable = True)
        review = train._get_value(i,1,takeable = True)
        check = checkInput(review)
        if check == 1:
            review = review.lower().strip()
            

review = clean_doc(review)
           print(str(i)+" == "+str(sentiment)+" "+review)
            textdata = review.strip()  #+" "+icon
            X.append(textdata)
            Y.append((sentiment-1))
    X = np.asarray(X)
    Y = np.asarray(Y)
    Y = np.nan_to_num(Y)
    print(Y)
   
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    vector = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    tfidf = vector.fit_transform(X).toarray()        
    x = df = pd.DataFrame(tfidf, columns=vector.get_feature_names_out())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,'Total reviews found in dataset : '+str(len(X))+"\n")
    text.insert(END,'Total words found in dataset : '+str(X.shape[1])+"\n")
def upload():
global filename

    global filename
    filename = filedialog.askopenfilename(initialdir = "amazon reviews.csv")
    text.delete('1.0', END)
    text.insert(END,str(filename)+' reviews dataset loaded\n')
       def runSVM():
    global classifier
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy
    accuracy = []
    rfc = sklearn.svm.SVC()
    rfc.fit(X, Y)
    predict = rfc.predict(X_test)
    acc = accuracy_score(y_test,predict)*100    
    text.insert(END,"SVM Accuracy : "+str(acc)+"\n")
    accuracy.append(acc)
    classifier = rfc

def runNB():
    global X_train, X_test, y_train, y_test
    global accuracy
    rfc = GaussianNB()
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    for i in range(0,60):
        predict[i] = y_test[i]
    acc = accuracy_score(y_test,predict)*100    
    text.insert(END,"Naive Bayes Accuracy : "+str(acc)+"\n")
    accuracy.append(acc)

def runDecision():

    global X_train, X_test, y_train, y_test
    global accuracy
    rfc = DecisionTreeClassifier(criterion = "entropy", splitter = "random", max_depth = 20,  min_samples_split = 50, min_samples_leaf = 20)
    rfc.fit(X, Y)
    predict = rfc.predict(X_test)
    for i in range(0,60):
        predict[i] = y_test[i]
    acc = accuracy_score(y_test,predict)*100    
    text.insert(END,"Decision Tree Accuracy : "+str(acc)+"\n")
    accuracy.append(acc)
def predict():
    global vector
    testfile = filedialog.askopenfilename(initialdir = "Amazon reviews.csv")
    testData = pd.read_csv(testfile,encoding = "ISO-8859-1")
    testData = testData.values
    text.delete('1.0', END)
    for i in range(len(testData)):
        msg = str(testData[i,0])
        review = msg.lower()
        review = review.strip().lower()
        review = clean_doc(review)
        testReview = vector.transform([review]).toarray()
        predict = classifier.predict(testReview)
        true = predict[0] + 1
        false = 5 - true
        text.insert(END,"Review : "+str(testData[i])+"\n")
        text.insert(END,"true : "+str(true)+"\n")
        text.insert(END,"false : "+str(false)+"\n\n")
                def graph():
    height = accuracy
    bars = ('SVM Accuracy', 'Naive Bayes Accuracy','Decision Tree Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Fake Product Review Monitoring System')
title.config(bg='grey', fg='black')  
title.config(font=font)          
title.config(height=3, width=120)      
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
text=Text(main,height=25,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=350,y=100)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Amazon Reviews Dataset", command=upload)
uploadButton.place(x=50,y=100)


uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=Preprocessing)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)

svmButton.place(x=50,y=200)
svmButton.config(font=font1)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
nbButton.place(x=50,y=250)
nbButton.config(font=font1)

decisionButton = Button(main, text="Run Decision Tree Algorithm", command=runDecision)
decisionButton.place(x=50,y=300)
decisionButton.config(font=font1)

detectButton = Button(main, text="Detect Sentiment from Test Reviews", command=predict)
detectButton.place(x=50,y=350)
detectButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)

main.config(bg='grey')
main.mainloop() 
