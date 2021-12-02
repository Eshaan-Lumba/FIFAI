from io import BytesIO
from fastai.vision.all import *
import requests
import streamlit as st
import sys

# save training and inference  models to specific files
# export model with fastai .pkl
# import it using streamlit
# save paramter in inference and trianing, we could up csv with results

st.title("Premier League Match Predictor")
home_team = st.text_input("Select Home team: ")

# if(st.button('Submit')):
#     result = home_team.title()
#     st.success(result)

away_team = st.text_input("Select Away team: ")

# if(st.button('Submit')):
#     away_result = away_team.title()
#     st.success(away_result)

home_streak_status = st.radio("Home 3 Game Win-Streak: ", ('Yes', 'No'))
 
# conditional statement to print
# Male if male is selected else print female
# show the result using the success function
# if (home_streak_status == 'Yes'):
#     st.success("Yes")
# else:
#     st.success("No")

away_streak_status = st.radio("Away 3 Game Win-Streak: ", ('Yes', 'No'))
 
# conditional statement to print
# Male if male is selected else print female
# show the result using the success function
# if (away_streak_status == 'Yes'):
#     st.success("Yes")
# else:
#     st.success("No")

home_win_streak_length = st.text_input("Enter current win-streak of home team:")

# if(st.button('Submit')):
#     streak = home_win_streak_length.title()
#     st.success(streak)

away_win_streak_length = st.text_input("Enter current win-streak of away team:")

# if(st.button('Submit')):
#     away_streak = away_win_streak_length.title()
#     st.success(away_streak)

referee = st.text_input("Enter referee of match: ")

# def predict(img):
#     st.image(img, caption="Your image", use_column_width=True)
#     pred, _, probs = learn_inf.predict(img)

#     f"""
#     prediction = {pred} with a probability of {probs.max()}%
#     probabilities = {probs}
#     """


# path = Path(sys.argv[1])
# learn_inf = load_learner(path)

# option = st.radio("", ["Upload Image", "Image URL"])

# if option == "Upload Image":
#     uploaded_file = st.file_uploader("Please upload an image.")

#     if uploaded_file is not None:
#         img = PILImage.create(uploaded_file)  # type: ignore
#         predict(img)

# else:
#     url = st.text_input("Please input a url.")

#     if url != "":
#         try:
#             response = requests.get(url)
#             pil_img = PILImage.create(BytesIO(response.content))  # type: ignore
#             predict(pil_img)

#         except:
#             st.text("Problem reading image from", url)  # type: ignore