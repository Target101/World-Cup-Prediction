#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

from xgboost import XGBRegressor


# In[119]:


d = pd.read_csv('data/players_20.csv')
d.shape


# In[120]:


d.head()


# In[121]:


l = list(d.columns)


# In[ ]:





# In[122]:


di = dict(d.isnull().sum())
null_list = []
for i in di:
    try:
        if di[i] > 0:
            null_list.append(i)
    except:
        continue
st.text(null_list)


# In[123]:


#st.text(l[45:])


# In[124]:


d['player_traits']=d['player_traits'].replace(np.nan, 'Unknown')


# In[125]:


d['team_jersey_number']= d['team_jersey_number'].dropna


# In[126]:


d['pace'] = d['pace'].fillna(0)


# In[127]:


for i in null_list[:8]:
    d[i] = d[i].replace(np.nan,'Unknown')


# In[128]:


for i in null_list[9:21]:
    d[i] = d[i].replace(np.nan,0)


    


# In[129]:


#x = '93+2'
#x[:2]


# In[130]:


def total_score(x):
    if x == '0':
        return int(x)
    y = int(x[:2])
    z = int(x[-1])
    c = y+z
    return c


# In[131]:


#null_list


# In[132]:


for i in null_list[22:]:
    d[i] = d[i].replace(np.nan,'0')


# In[133]:


for i in null_list[22:]:
    d[i] = d[i].apply(total_score)


# In[134]:


#for i in d['cam']:
    #st.text(type(i))


# In[135]:


#d['ls'].unique()


# In[136]:


#d['ls']


# In[137]:
st.dataframe(d.head())


# In[138]:


#d['player_traits']


# In[ ]:





# In[139]:


#d['overall']


# In[140]:


d['player_traits'] = d['player_traits'].replace(np.nan, 'unknown')
#d['player_traits']
d['final_trait'] =d['player_traits'] + d['player_positions']

# In[141]:


#d.isnull().sum()


# In[142]:


v = d


# In[143]:


#v.head()


# In[144]:


corr_matrix = v.corr()
sns.heatmap(corr_matrix, annot=True)



# In[145]:

st.markdown("Finding correlation between all the features")
st.write(corr_matrix. style. background_gradient (cmap = 'BrBG')  )
fig, ax = plt.subplots()
sns.heatmap(d.corr(), ax=ax)
st.write(fig)


# In[146]:


v = v.drop(['player_url','sofifa_id','height_cm','weight_kg'],axis =1 )


# In[147]:


v = v.drop(['nation_jersey_number'],axis =1)


# In[148]:


v = v.drop(['dob'],axis = 1)


# In[149]:


v = v.drop(['real_face'],axis =1 )


# In[150]:


#v.columns


# In[151]:


v['attack_score'] = (v['attacking_crossing'] + v['attacking_finishing'] + v['attacking_heading_accuracy'] + v['attacking_short_passing'] + v['attacking_volleys'])/5


# In[152]:


v['skill_score'] = (v['skill_ball_control']+v['skill_curve']+v['skill_dribbling'] + v['skill_fk_accuracy'] + v['skill_moves'] + v['skill_long_passing'])/6


# In[153]:


v['movement_score'] = (v['movement_acceleration']+v['movement_balance']+v['movement_agility']+v['movement_sprint_speed']+v['movement_reactions'])/5


# In[154]:


v['power_score'] = (v['power_jumping']+v['power_long_shots']+v['power_shot_power']+v['power_stamina']+v['power_strength'])/5


# In[155]:


v['mentality_score'] = (v['mentality_aggression']+v['mentality_composure']+v['mentality_interceptions']+v['mentality_penalties']+v['mentality_positioning']+v['mentality_vision'])/6


# In[156]:


v['defending_score'] =  (v['defending_marking']+v['defending_sliding_tackle']+v['defending_standing_tackle'])/3


# In[157]:


v['goalie_score'] = (v['goalkeeping_diving']+v['goalkeeping_handling']+v['goalkeeping_kicking']+v['goalkeeping_positioning']+v['goalkeeping_reflexes'])/5


# In[158]:


#v['power_score']


# In[159]:


#v['skill_score']


# In[160]:


#v['defending_score']


# In[161]:


#v['goalkeeping_diving']


# In[ ]:





# In[162]:


v = v.drop(['attacking_crossing','attacking_finishing','attacking_heading_accuracy','attacking_short_passing','attacking_volleys'],axis = 1)


# In[163]:


v = v.drop (['skill_ball_control','skill_curve','skill_dribbling','skill_fk_accuracy','skill_moves','skill_long_passing'],axis=1)


# In[164]:


v = v.drop(['movement_acceleration','movement_balance','movement_agility','movement_sprint_speed','movement_reactions'],axis =1 )


# In[165]:


v = v.drop(['power_jumping','power_long_shots','power_shot_power','power_stamina','power_strength'],axis = 1)


# In[166]:


v = v.drop(['mentality_aggression','mentality_composure','mentality_interceptions','mentality_penalties','mentality_positioning','mentality_vision'],axis =1 )


# In[167]:


v = v.drop(['defending_marking','defending_sliding_tackle','defending_standing_tackle'],axis = 1)


# In[168]:


#v.columns


# In[169]:


#v['body_type'].unique()


# In[170]:


v = v.drop(['body_type'],axis =1 )


# In[ ]:





# In[171]:


#v['club'].unique()


# In[172]:

st.markdown("Label encoding of all categorical variables ")
u = v
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
u['club']= label_encoder.fit_transform(u['club'])
  
#st.table(u['club'].unique())


# In[173]:



label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
u['nationality']= label_encoder.fit_transform(u['nationality'])
  
#st.table(u['nationality'].unique())


# In[174]:


label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
u['player_positions']= label_encoder.fit_transform(u['player_positions'])
  
#st.table(u['player_positions'].unique())


# In[175]:


#v['player_traits']


# In[176]:


#u.columns


# In[177]:


u = u.drop(['player_traits','joined','player_tags','contract_valid_until','nation_position'],axis = 1)


# In[178]:


#u.columns


# In[179]:


u.columns


# In[180]:


#u['team_position']


# In[181]:


label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
u['team_position']= label_encoder.fit_transform(u['team_position'])
  
#st.table(u['team_position'].unique())




# In[182]:


#u['preferred_foot'] = label_encoder.fit_transform(u['preferred_foot'])


# In[183]:


u = u.drop(['team_jersey_number','loaned_from','gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes',
       'gk_speed', 'gk_positioning'],axis = 1)


# In[184]:


#u.columns


# In[185]:


st.write(u.corr(). style. background_gradient (cmap = 'BrBG')  )


# In[186]:


u = u.drop(['player_positions','preferred_foot'],axis =1 )


# In[187]:


#u['preffered_foot'] = v['preferred_foot']
#u['preferred_foot']= label_encoder.fit_transform(u['prefferred_foot'])


# In[188]:


label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
u['work_rate']= label_encoder.fit_transform(u['work_rate'])
  
#st.table(u['work_rate'].unique())


# In[189]:


X = u.drop(['short_name','long_name','ls', 'st', 'rs',
       'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
       'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb',
       'rcb', 'rb','overall','final_trait'],axis =1 )


# In[190]:


#X.columns
y = u['overall']


# In[191]:


#X = X.drop(['attacking_crossing', 'attacking_finishing',
       #'attacking_heading_accuracy', ',
       #'attacking_volleys'],axis=1)

#X.columns


# In[192]:

st.header("Using models to predict the ratings of players ")
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,shuffle= True, random_state = 42)


# In[193]:


X['release_clause_eur'] = X['release_clause_eur'].replace('Unknown',sum(X['value_eur']+X['wage_eur']))
X = X.drop(['team_position'],axis=1)


# In[195]:


#X= X.drop(['preffered_foot'],axis =1 )


# In[196]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,shuffle= True, random_state = 42)


# In[197]:
st.markdown("RandomForestModel")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
regressor = RandomForestRegressor(n_estimators = 300,random_state = 41)
regressor.fit(X_train,y_train)
y_test_pred = regressor.predict(X_test)

st.text('-'*70)
st.text('Random Forest Regression model')
st.text('-'*70)
st.text(f'train score : {regressor.score(X_train, y_train)}')
st.text(f'test score  : {regressor.score(X_test, y_test)}')
st.text('-'*70)
st.text(f'r2 score for test  : {r2_score(y_test, y_test_pred)}')




from sklearn import metrics

st.text(f'MAE:{metrics.mean_absolute_error(y_test, y_test_pred)}')
st.text(f'MSE:{metrics.mean_squared_error(y_test, y_test_pred)}')
st.text(f'RMSE:{np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))}')


st.markdown("XGBOOST MODEL")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
xgb = XGBRegressor()
xgb.fit(X_train,y_train)
y_test_pred = xgb.predict(X_test)

st.text('-'*70)
st.text('')
st.text('-'*70)
st.text(f'train score : {xgb.score(X_train, y_train)}')
st.text(f'test score  : {xgb.score(X_test, y_test)}')
st.text('-'*70)
st.text(f'r2 score for test  : {r2_score(y_test, y_test_pred)}')



from sklearn import metrics

st.text(f'MAE:{metrics.mean_absolute_error(y_test, y_test_pred)}')
st.text(f'MSE:{metrics.mean_squared_error(y_test, y_test_pred)}')
st.text(f'RMSE:{np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))}')

st.markdown("top 5 attackers")
st.dataframe(v.sort_values("attack_score",ascending = False).head())
st.markdown("top 5 defenders")
st.dataframe(v.sort_values("defending_score",ascending = False).head())
st.markdown("top 5 goalkeepers")
st.dataframe(v.sort_values("goalie_score",ascending=False).head())
st.markdown("top 5 showboaters")
st.dataframe(v.sort_values("skill_score",ascending=False).head())
st.markdown("top 5 passers")
st.dataframe(v.sort_values("passing",ascending=False).head())
st.markdown("top 5 dribblers")
st.dataframe(v.sort_values("dribbling",ascending=False).head())

# # COSINE SIMILARITY



name = list(d['long_name'])
trait = list(d['final_trait'])
res = {}
for key in name :
    for j in trait:
        res[key] = j
        trait.remove(j)
        break
    



def similarity(new_data,s,tr):
    similar = []
    vectorizer = TfidfVectorizer()
    for i in new_data:
        
            if i == s:
                continue
            corpus = [new_data[i],tr]
            trsfm=vectorizer.fit_transform(corpus)
            doc1 = trsfm[0:1].todense()
            doc2 = trsfm[1:2].todense()
            doc_1 = []

            for k in range(len(doc1[0])):
                doc_1.append(doc1[k])
            doc_2 = []
            for u in range(len(doc2[0])):
                doc_2.append(doc2[u])
            doc_1 = np.squeeze(np.asarray(doc_1))
            doc_2 = np.squeeze(np.asarray(doc_2))
            # Dot and norm
            dot = sum(a*b for a, b in zip(doc_1, doc_2))
            norm_a = sum(a*a for a in doc_1) ** 0.5
            norm_b = sum(b*b for b in doc_2) ** 0.5

            # Cosine similarity
            cos_sim = dot / (norm_a*norm_b)
            #st.text([i,j,cos_sim])
            
            similar.append([i,s,cos_sim])
    return similar

c = 1
s = ""
pos = ""
tr = ""


s = st.text_input("Enter player name")
pos = st.text_input("Enter the positions of that player")
tr = st.text_input("Enter traits of that player")
tr = tr + " " + pos
    
similar = similarity(res,s,tr)
cols = ['Name1','Name2','similarity']
s_df = pd.DataFrame(similar,columns=cols)


st.markdown("The 5 players who are very similar to this player are ")
sorted_sdf = s_df.sort_values('similarity',ascending=False)
st.dataframe(sorted_sdf.head())


