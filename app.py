import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import numpy as np
import spacy
from spacy.cli.download import download
download(model="en_core_web_sm")

nlp = spacy.load('en_core_web_sm')




url='https://raw.githubusercontent.com/thecodebloodedguy/career-guidance-ml-model/main/dataset-github.csv'
df=pd.read_csv(url)


q_counter=0
flag=0
q_list=["What is your percentage in Operating System?",
"What is your percentage in Algorithms?","What is your percentage in Programming Concepts?",
"What is your percentage in Software Engineering?","What is your percentage in Computer Networks?",
"What is your percentage in Electronics Subjects",
"What is your percentage in Computer Architecture",
"What is your percentage in Mathematics",
"What is your percentage in  Communication skills",
"How many hours can you work in a day?",
"How much would you rate your Logical Quotient from 1-10?",
"How many hackathons have you participated in?",
"How much would you rate your coding skills from 1-10?",
"How much would you rate your public speaking skills from 1-10?",
"Can you work in front of a computer?(yes or no)",
"Do you have self-learning capability?(yes or no)",
"What extra courses have you done?",
"What type of workshops have you attended?",
"Which subject are you mostly interested in?",
"What are your career interests?",
"Do you want to choose job or higher studies?",
"What type of company would you like to settle in?",
"Do you want to work in management or techincal sector?",
"Have you ever worked in teams?"]

a=[]

#data modifications
df["certifications"]=df['certifications'].replace(['machine learning', 'hadoop', 'python', 'information security', 'app development', 'shell programming', 'r programming', 'distro making', 'full stack'],[1,2,3,4,5,6,7,8,9])
df['workshops']=df['workshops'].replace(['data science', 'system designing', 'testing', 'game development', 'web technologies', 'database security', 'hacking', 'cloud computing'],[1,2,3,4,5,6,7,8])
df['Interested subjects']=df['Interested subjects'].replace(['IOT', 'programming', 'Computer Architecture', 'Management', 'Software Engineering', 'hacking', 'parallel computing', 'cloud computing', 'networks', 'data engineering'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df['interested career area ']=df['interested career area '].replace(['testing', 'system developer', 'security', 'Business process analyst', 'developer', 'cloud computing'],[1, 2, 3, 4, 5, 6])
df['Type of company want to settle in?']=df['Type of company want to settle in?'].replace(['SAaS services', 'Product based', 'Service Based', 'BPA', 'Cloud Services', 'Web Services', 'Finance', 'Testing and Maintainance Services', 'Sales and Marketing', 'product development'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df['Suggested Job Role']=df['Suggested Job Role'].replace(['Database Developer', 'Network Security Engineer', 'CRM Technical Developer', 'Solutions Architect', 'Technical Services/Help Desk/Tech Support', 'Technical Engineer', 'UX Designer', 'CRM Business Analyst', 'Mobile Applications Developer', 'Software Engineer', 'Programmer Analyst', 'E-Commerce Analyst', 'Portal Administrator', 'Software Quality Assurance (QA) / Testing', 'Software Developer', 'Web Developer', 'Database Administrator', 'Data Architect', 'Business Systems Analyst', 'Database Manager', 'Quality Assurance Associate', 'Design & UX', 'Project Manager', 'Systems Analyst', 'Applications Developer', 'Network Engineer', 'Information Technology Auditor', 'Information Technology Manager', 'Software Systems Engineer', 'Network Security Administrator', 'Information Security Analyst', 'Technical Support', 'Business Intelligence Analyst', 'Systems Security Administrator'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
df['Job/Higher Studies?']=df['Job/Higher Studies?'].replace(['job', 'higherstudies'],[2,3])
df['Management or Technical']=df['Management or Technical'].replace(['Technical', 'Management'],[2,3])
df["worked in teams ever?"]=df["worked in teams ever?"].replace(["yes","no"],[1,0])
df["self-learning capability?"]=df["self-learning capability?"].replace(["yes","no"],[1,0])
df["can work long time before system?"]=df["can work long time before system?"].replace(["yes","no"],[1,0])

#ml model train
X = df.drop(['Suggested Job Role'], axis=1)
Y=df['Suggested Job Role']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
rf_m = RandomForestClassifier(max_features=2, random_state=42, oob_score=True)
rf_m.fit(X_train, Y_train)

#chatbot preparation
bot = ChatBot('Bot',logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.TimeLogicAdapter'])
  
trainer = ListTrainer(bot)
  
trainer.train([
    'Hi',
    'Hello',
    'I need roadmap for Competitive Programming',
    'Just create an account on GFG and start',
    'I have a query.',
    'Please elaborate, your concern',
    'How long it will take to become expert in Coding ?',
    'It usually depends on the amount of practice.',
    'Ok Thanks',
    'No Problem! Have a Good Day!'
])
  
#cat codes coversions

def certification(n):
 
 
      if n=="machine learning".lower():
          return 1
      elif n=="hadoop".lower():
          return 2
      elif n=="python".lower():
          return 3
      elif n=="information security".lower():
          return 4
      elif n=="app development".lower():
        return 5
      elif n=="shell programming".lower():
        return 6
      elif n=="r programming".lower():
        return 7
      elif n=="distro making".lower():
        return 8 
      elif n=="full stack".lower():
        return 9
      else:
        return -1
		

def workshop(n):
 
 
      if n=="data science".lower():
          return 1
      elif n=="system designing".lower():
          return 2
      elif n=="testing".lower():
          return 3
      elif n=="game development".lower():
          return 4
      elif n=="web technologies".lower():
        return 5
      elif n=="database security".lower():
        return 6
      elif n=="hacking".lower():
        return 7
      elif n=="cloud computing".lower():
        return 8 
      else:
        return -1
		
def interested_subject(n):
 
 
      if n=="IOT".lower():
          return 1
      elif n=="programming".lower():
          return 2
      elif n=="Computer Architecture".lower():
          return 3
      elif n=="Management".lower():
          return 4
      elif n=="Software Engineering".lower():
        return 5
      elif n=="hacking".lower():
        return 6
      elif n=="parallel computing".lower():
        return 7
      elif n=="cloud computing".lower():
        return 8 
      elif n=="networks".lower():
        return 9
      elif n=="data engineering".lower():
        return 10     
      else:
        return -1
		
def interested_career_area(n):
 
 
      if n=="testing".lower():
          return 1
      elif n=="system developer".lower():
          return 2
      elif n=="security".lower():
          return 3
      elif n=="Business process analyst".lower():
          return 4
      elif n=="developer".lower():
        return 5
      elif n=="cloud computing".lower():
        return 6
      else:
        return -1
		
def company_type(n):
 
 
      if n=="SAaS services".lower():
          return 1
      elif n=="Product based".lower():
          return 2
      elif n=="Service Based".lower():
          return 3
      elif n=="BPA".lower():
          return 4
      elif n=="Cloud Services".lower():
        return 5
      elif n=="Web Services".lower():
        return 6
      elif n=="Finance".lower():
        return 7
      elif n=="Testing and Maintenance Services".lower():
        return 8 
      elif n=="Sales and Marketing".lower():
        return 9
      elif n=="product development".lower():
        return 10     
      else:
        return -1
		
def job_role(n):
 
 
      if n==1:
          return "Database Developer"
      elif n==2:
          return "Network Security Engineer"
      elif n==3:
          return "CRM Technical Developer"
      elif n==4:
          return "Solutions Architect"
      elif n==5:
        return "Technical Services/Help Desk/Tech Support"
      elif n==6:
        return "Technical Engineer"
      elif n==7:
        return "UX Designer"
      elif n==8:
        return "CRM Business Analyst"
      elif n==9:
        return "Mobile Applications Developer"
      elif n==10:
        return "Software Engineer"
      elif n==11:
        return "Programmer Analyst"
      elif n==12:
        return "E-Commerce Analyst"
      elif n==13:
        return "Portal Administrator"
      elif n==14:
        return "Software Quality Assurance (QA) / Testing"
      elif n==15:
        return "Software Developer"
      elif n==16:
        return "Web Developer"
      elif n==17:
        return "Database Administrator"
      elif n==18:
        return "Data Architect"
      elif n==19:
        return "Business Systems Analyst"
      elif n==20:
        return "Database Manager"
      elif n==21:
        return "Quality Assurance Associate"
      elif n==22:
        return "Design & UX"    
      elif n==23:
        return "Project Manager"
      elif n==24:
        return "Systems Analyst"
      elif n==25:
        return "Applications Developer"
      elif n==26:
        return "Network Engineer"
      elif n==27:
        return "Information Technology Auditor"
      elif n==28:
        return "Information Technology Manager"     
      elif n==29:
        return "Software Systems Engineer"
      elif n==30:
        return "Network Security Administrator"
      elif n==31:
        return "Information Security Analyst"
      elif n==32:
        return "Technical Support"   
      elif n==33:
        return "Business Intelligence Analyst"
      elif n==34:
        return "Systems Security Administrator"   
      else:
        return -1
		
def job_hs(n):
 
 
      if n=="job".lower():
          return 2
      elif n=="higherstudies".lower():
          return 3
      else:
        return -1
		
def mng_tech(n):
 
 
      if n=="Technical".lower():
          return 2
      elif n=="Management".lower():
          return 3
      else:
        return -1


#questionnaire get_response() function


def get_response_(sentence):
    global q_counter
    global q_list
    global a
    global flag
    if(q_counter==0):
        flag=1
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==1):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==2):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==3):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==4):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==5):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==6):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==7):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==8):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==9):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==10):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==11):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==12):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==13):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==14):
        a.append(int(sentence))
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==15):
        res=1 if sentence.lower()=="yes" else 0
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==16):
        res=1 if sentence.lower()=="yes" else 0
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==17):
        res=certification(sentence.lower())
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==18):
        res=workshop(sentence.lower())
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==19):
        res=interested_subject(sentence.lower())
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==20):
        res=interested_career_area(sentence.lower())
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==21):
        res=job_hs(sentence.lower())
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==22):
        res=company_type(sentence.lower())
        a.append(res)
        q_counter+=1
        return q_list[q_counter-1]
    if(q_counter==23):
        res=mng_tech(sentence.lower())
        a.append(res)
        q_counter+=1
        flag=2
        return q_list[q_counter-1]
    if(flag==2):
        res=1 if sentence.lower()=="yes" else "no"
        a.append(res)
        q_counter+=1
        flag=0
        a=np.array(a)
        a=a.reshape(1, -1)
        result=rf_m.predict(a)
        return job_role(result)
    
#running chatbot
#  
while True:
    request=input('You :')
    if request.lower() == 'quit':
        print('Bot: bye')
        break
    elif (request.lower()=="get career guidance" or flag>=1):
      response=get_response_(request)
      print('Bot:',response)
    else:
        response=bot.get_response(request)
        print('Bot:', response)


