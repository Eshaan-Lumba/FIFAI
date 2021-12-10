
from fastai.tabular.all import *
import streamlit as st


# +
"""
# Welcome to the FIFAI Premier League match predictor
"""

home_team = st.text_input("Enter the Home Team")
away_team = st.text_input("Enter the Away Team")
home_win_streak = float(st.slider("Enter the winning streak of the home team", 0, 11))
away_win_streak = float(st.slider("Enter the winning streak of the away team", 0, 11))
TGSH = st.radio('Is the home team on a 3 game winning streak?', ['Yes', 'No'])
is_on_TGSH = 0.0
if (TGSH == 'Yes'):
    is_on_TGSH = 1.0
TGSA = st.radio('Is the away team on a 3 game winning streak?', ['Yes', 'No'])
is_on_TGSA = 0.0
if (TGSA == 'Yes'):
    is_on_TGSA = 1.0
referee = st.text_input("Enter the referee")


# -


def load_model():
    path = 'final_model.pkl'
    learn = load_learner(path, 'final_model.pkl')
    return learn


mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]

mydict = [{'HomeTeam': home_team, 'AwayTeam': away_team, 'HomeWinStreak': home_win_streak,
           'AwayWinStreak': away_win_streak,  'TGSH': is_on_TGSH, 'TGSA': is_on_TGSA, 'Referee': referee}]
df = pd.DataFrame(mydict)
predict_input = df.iloc[0]

learn = load_model()
row, clas, probs = learn.predict(predict_input)

# +
LDW = torch.argmax(probs)

if (LDW == 0):
    st.subheader(f"Our model predicts a win for {away_team}")
elif (LDW == 1):
    st.subheader(f"Our model predicts a draw between {home_team} and {away_team}")
else:
    st.subheader(f"Our model predicts a win for {home_team}")
# -


print("Running")
