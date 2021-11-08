#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd 


# In[52]:


# open data files
Sixteen_Seventeen_data = pd.read_csv("./Desktop/CS152/16_17_epl_data.csv", sep = ",")
Seventeen_Eighteen_data = pd.read_csv("./Desktop/CS152/17_18_epl_data.csv", sep = ",")
Eighteen_Nineteen_data = pd.read_csv("./Desktop/CS152/18_19_epl_data.csv", sep = ",")
Nineteen_Twenty_data = pd.read_csv("./Desktop/CS152/19_20_epl_data.csv", sep = ",")
Twenty_TwentyOne_data =  pd.read_csv("./Desktop/CS152/20_21_epl_data.csv", sep = ",")


# In[53]:


Sixteen_Seventeen_data.head()


# In[54]:


# Look through the first row to get column names 
list_of_column_names = []
# loop to iterate through the rows of csv
for row in Sixteen_Seventeen_data:
    list_of_column_names.append(row)
print(list_of_column_names)


# In[55]:


#Put all the data files into a list
seasons_data_list = [Sixteen_Seventeen_data, Seventeen_Eighteen_data, Eighteen_Nineteen_data, Nineteen_Twenty_data, Twenty_TwentyOne_data]


# In[56]:


# Function to extract the needed column data
# i.e "HomeTeam", "AwayTeam", "Full time home goals","Full time away goals", "Results"
def get_relevant_data(team_name, data):
    relevant_data = [["HomeTeam", "AwayTeam", "Full time home goals","Full time away goals", "Results" ]]
    for i, row in data.iterrows():
        if (row["HomeTeam"] == team_name) or (row["AwayTeam"] == team_name):
            game_entry = []
            game_entry.append(row["HomeTeam"])
            game_entry.append(row["AwayTeam"])
            game_entry.append(row["FTHG"])
            game_entry.append(row["FTAG"])
            if row["HomeTeam"] == team_name :
                results = int(row["FTHG"]) - int(row["FTAG"])
            else:
                results = int(row["FTAG"]) - int(row["FTHG"])    
            game_entry.append(results)
            relevant_data.append(game_entry)
    return relevant_data


# In[57]:


# Function to calculate and create a win streak column
def create_win_streak(team_data):
    team_data[0].append("Win Streak")
    team_data[0].append("3 Game Streak")
    win_streak = 0
    three_game_streak = 0 
    for i in range(1, len(team_data)):
        if team_data[i][4] > 0:
            win_streak += 1
        else:
            win_streak = 0
            
        if win_streak < 3:
            three_game_streak = 0
        else: 
            three_game_streak = 1   
        team_data[i].append(win_streak)
        team_data[i].append(three_game_streak)
        
    return team_data


# In[58]:


# find non-relegation teams
def team_list_per_season(season_data):
    team_list = []
    for i, row in season_data.iterrows():
        team_list.append(row["HomeTeam"])
    return set(team_list)

# create team lists by season
team_list_by_season = []
for season in seasons_data_list:
    team_list_by_season.extend(team_list_per_season(season))
    
from collections import Counter 
counter_map = Counter(team_list_by_season)

non_relegated_list = []
for key, value in counter_map.items():
    # check if a team has been in the league for all the seasons we are using the data for
    if value == len(seasons_data_list):
        non_relegated_list.append(key)
        
#print(non_relegated_list)
#DATA USED
#Fifteen_Sixteen_data (removed)
#Sixteen_Seventeen_data
#Seventeen_Eighteen_data 
#Eighteen_Nineteen_data
#Nineteen_Twenty_data
#Twenty_TwentyOne_data 


# In[59]:


#create csv files for all the non-relegated teams for the last 5 seasons
for team in non_relegated_list:
    six_Seventeen_team_data = pd.DataFrame(create_win_streak(get_relevant_data(team, Sixteen_Seventeen_data)))
    six_Seventeen_team_data.to_csv("./Desktop/CS152/team_data/" + team + "16_17_data.csv",index=True, header=True)
    
    seven_Eighteen_team_data = pd.DataFrame(create_win_streak(get_relevant_data(team, Seventeen_Eighteen_data)))
    seven_Eighteen_team_data.to_csv("./Desktop/CS152/team_data/" + team + "17_18_data.csv",index=True, header=True)
    
    eight_Nineteen_team_data = pd.DataFrame(create_win_streak(get_relevant_data(team, Eighteen_Nineteen_data)))
    eight_Nineteen_team_data.to_csv("./Desktop/CS152/team_data/" + team + "18_19_data.csv",index=True, header=True)
    
    nine_Twenty_team_data = pd.DataFrame(create_win_streak(get_relevant_data(team, Nineteen_Twenty_data)))
    nine_Twenty_team_data.to_csv("./Desktop/CS152/team_data/"   + team + "19_20_data.csv",index=True, header=True)
    
    twenty_TwentyOne_team_data = pd.DataFrame(create_win_streak(get_relevant_data(team, Twenty_TwentyOne_data)))
    twenty_TwentyOne_team_data.to_csv("./Desktop/CS152/team_data/" + team + "20_21_data.csv",index=True, header=True)
    


# In[ ]:




