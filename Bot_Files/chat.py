import random
import json
import pickle
import torch
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
filename = 'finalized_model.sav'
bot_name = "PharmDroid"
rf = pickle.load(open(filename, 'rb'))


q_counter=0
flag=0
q_list =  [
        "What is your stream?",
        "What is your gender?",
        "On a scale of 0-10,how much would you rate your visualisation skills?",
        "On a scale of 0-10,how much would you rate your logical quotient skills?",
        "On a scale of 0-10,how much would you rate your reading skills?",
        "What are your marks in Higher Secondary?",
        "Are you interested in the field of sports?",
        "What are your accomplishments in sports?",
        "Are you interested in the field of arts?",
        "What type of art field are you interested in?",
        "On a scale of 0-10,how much would you rate your teaching skills?",
        "On a scale of 0-10,how much you rate your communication skills?",
        "Would you say that you are good on stage?",
        "Can you lead a team?",
        "On a scale of 0-10,how much would you rate your team management skills?",
        "Are you interested in the field of Chemistry?",
        "On a scale of 0-10,how much would you rate your chemistry skills?"
        "Are you interested in the field of Tech?",
        "What domain of Tech are you interested in, hardware or software?",
        "On a scale of 0-10,how much would you rate your coding skills?",
        "On a scale of 0-10,how much would you rate your technical skills?",
        "Are you interested in the field of Biology?",
        "What field of Biology are you interested in?",
        "What are your preferences in the Biology sub-fields?",
        "Do you have a family business?",
        "What type is your family business?",
        "On a scale of 0-10,how much would you rate your interest in doing public services?",
        "On a scale of 0-10,how much would you rate your Mathematics skills?",
        "On a scale of 0-10,how much would you rate your visualization skills?"

    ]

q_res=[["Don't worry at all, I've still got your back","Sit back and relax, science students like you have a vast variety of options","Backbone of a company's Accounting. You're in the game!","The lifeline of our bodies! You are in for big.","Looks like you are following your passion, let's se what I have for you."],
         ["Hello Mister","Hello Miss"],
         ["Looks like your brain needs still more logical thinking approach","I can tell that you have mastered the logical quizzes"],
         ["Looks like you don't love books that much","I smell an avid reader here"],
         ["Looks like you had a tough time","I can say that you are a hard worker","I can smell a topper from a mile away"],
         ["Not a sports guy, huh","Now we are talking!"],
         ["I can tell you must be pretty good at sports","Now that's a real sportsperson!"],
         ["You are in for some good options","Looks like your interest is not in Arts"],
         ["Oh! I love that field"],
         ["I mean if everyone is a teacher, who will be the student?","I can tell,you can become a great teacher"],
         ["Looks like you get your work done all by yourself","Looks like you are sound"],
         ["Seems like you have stage fear","Looks like you are ready to perform on stage"],
         ["Not everyone likes to work on a team","You are fit to lead!"],
         ["Looks like you work your best when you work alone","I am sure you were the leader in school projects"],
         ["Looks like you might be a fan of Breaking Bad too","We still have much options left for you"],
         ["Chemistry is tough for me too","You have a way with handling chemicals"],
         ["You just opened a big pile of options for yourself","Don't worry there are many good non-techincal fields"],
         ["Let's see what I have further to offer you","Great choice! You are in for big time"],
         ["Looks like you nexzzzed more practice","I am sure you are good with keyboard"],
         ["You will get there, don't worry","Looks like you are sound"],
         ["You are in for big time","Not everyone likes it"],
         ["Awesome choice my friend!"],
         ["Now that is an amazing preference"],
         ["Cool! Looks like you might have it sorted","Working a 9-5 is good too"],
         ["Looks like this is not your thing","You sound like a kind hearted person"],
         ["Not everyone is good in maths","You have a way with numbers"],
         ["Looks like you still thrive for more visual skills","Looks like your brain is prepared to act in any situation"]
         ]

a=[0]*28

def get_response_(sentence):
    global q_counter
    global q_list
    global a
    global flag
    global q_res
    if(q_counter==0):
        flag=1
        q_counter+=1
        return "I will ask you a few questions. Answer me well \n"+q_list[q_counter-1]
    if(q_counter==1):
        res=q1(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==2):
        res=q2(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==3):
        res=q3(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==4):
        res=q4(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==5):
        res=q5(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==6):
        res=q6(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==7):
        res=q7(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==8):
        res=q8(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==9):
        res=q9(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==10):
        res=q10(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==11):
        res=q11(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==12):
        res=q12(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==13):
        res=q13(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==14):
        res=q14(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==15):
        res=q15(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==16):
        res=q16(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==17):
        res=q17(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==18):
        res=q18(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==19):
        res=q19(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==20):
        res=q20(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==21):
        res=q21(sentence)
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==22):
        res=q22(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==23):
        res=q23(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==24):
        res=q24(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==25):
        res=q25(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==26):
        res=q26(sentence)
        a[q_counter-1]=res
        q_counter+=1
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(q_counter==27):
        res=q27(sentence)
        a[q_counter-1]=res
        q_counter+=1
        flag=2
        return q_res[q_counter-1][res]+'\n'+q_list[q_counter-1]
    if(flag==2):
        res=q28(sentence)
        a[q_counter-1]=res
        q_counter+=1
        flag=0
        a=np.array(a)
        print(a)
        a=a.reshape(1, -1)
        result=rf.predict(a)
        print(result)
        return output(result[0])

def bot_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.25:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return f"{random.choice(intent['responses'])}"
    else:
        res=random.choice(['Can you explain it better?','Remember! I am not super smart','Say something in Layman Language'])
        return res

import regex as re
def chatter(sentence):
    global flag
    match = re.search(r'test', sentence)
    if sentence == "quit":     
        res="Byye!"
    elif match or flag>=1:
        res=get_response_(sentence)
    else:
        res=bot_response(sentence)
    return res