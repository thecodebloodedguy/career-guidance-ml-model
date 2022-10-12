import pandas as pd
import pickle
df=pd.read_csv('Bot_Files\Career_Data.csv')
df.fillna(0)
df['arts_rel'] = df['arts_rel'].replace(['0','music', 'theatre', 'tourism', 'fineart', 'literature'],[0,1,2,3,4,5])
df['tech_sh'] = df['tech_sh'].replace(['software', '0','hardware'],[1,0,2])
df['bio_class']=df['bio_class'].replace(['0','nutritionist', 'medical'],[0,1,2])
df['bio_pr']=df['bio_pr'].replace(['0','dental', 'orthopaedic', 'veterinary', 'pediatrician', 'physician'],[0,1,2,3,4,5])
df['fam_biz']=df['fam_biz'].replace(['0','construction', 'enterprise', 'industrial', 'agriculture'],[0,1,2,3,4,])
df['career']=df['career'].replace(['Marketing Management', 'Theatre & Tourism', 'Teaching', 'Medicine and Nutrition', 'Business/Industrial Mngmnt.', 'Graphics & Designing', 'UPSC/GPSC etc.', 'Chemical Engineering', 'ARTS', 'CA/CS/Banking', 'Sports Science'],[1,2,3,4,5,6,7,8,9,10,11])
df['tech_sh']=df['tech_sh'].replace(['software', 'hardware'],[1,2])
df['stream']=df['stream'].replace(['commerce','arts','maths','bio','none'],[3,4,2,1,0])
df['gender']=df['gender'].replace(['male','female','other'],[1,2,3])
df['sports_in']=df['sports_in'].replace(['yes','no'],[1,0])
df['arts_in']=df['arts_in'].replace(['yes','no'],[1,0])
df['bio_in']=df['bio_in'].replace(['yes','no'],[1,0])
df['fam_bus']=df['fam_bus'].replace(['yes','no'],[1,0])
df['tech_in']=df['tech_in'].replace(['yes','no'],[1,0])
df['chem_in']=df['chem_in'].replace(['yes','no'],[1,0])
df['stage_in']=df['stage_in'].replace(['yes','no'],[1,0])
X = df.drop(['career'], axis=1)
Y=df['career']
from sklearn.svm import SVC 
from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0).fit(X,Y)

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

#WARNING : The file is incomplete 
#proper model will be trained only after purified data from