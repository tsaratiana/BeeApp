import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import plotly.graph_objs as go
from streamlit.components.v1 import html
import streamlit_plotly_events as pse
import seaborn as sns
from geopy.geocoders import Nominatim


st.set_page_config(
    page_title="BeeApp",
    page_icon="🌻",
)


st.write('# 🐝 To bee or not to bee...')


st.markdown(
    """
    Bees are essential to our food system, with every third bite of food relying on their pollination.
    However, recent years have seen a dramatic decline in bee populations, leading to concerns about the future of
    our food supply. To better understand and protect these vital insects, we need to monitor their behavior and
    health more closely before it is too late. Traditionally, monitoring bees has been time-consuming and disruptive to the hive's workflow.

    The dataset used provides many bees information, such as pollen-carrying bees vs those without, paving the way
    for more intelligent hive monitoring or beekeeping. By exploring and cleaning this dataset,
    we can gain insights into bee behavior and characteristics,
    which can help inform better hive monitoring and beekeeping practices.

    Link to the source: [Here!](https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images?resource=download)

    ### Dataset
    - This dataset contains 5,100+ bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.
    - The images in this dataset were extracted from time-lapse videos of bees in motion.
    In order to isolate the bees, each frame of the video was subtracted against a background image obtained
    by averaging the frames. The bees were then cropped out of the frame to create images containing a single bee.
    The dataset is semi-automatically labeled using information provided in a form accompanying each video.
    Since the quality of the resulting image crops varies depending on the video, the labeling process is not entirely automated.
    As more videos and data become available, this dataset will be updated accordingly.
    - -1 means the information is coming soon.

    **👈 Click on the sidebar to explore the dataset!**  

"""
)

DATA_CSV = './data/bee_data.csv'
IMAGE_PATH = './data/bee_imgs/bee_imgs/'

@st.cache_data
def load_data():
    data = pd.read_csv('./data/bee_data.csv')
    return data

# Load the bee dataset
bee_df = load_data()

bee_df = bee_df.replace({'location':'Athens, Georgia, USA'}, 'Athens, GA, USA')

bee_df['date_time'] = pd.to_datetime(bee_df['date'] + ' ' + bee_df['time'])
bee_df["year"] = bee_df['date_time'].dt.year
bee_df["month"] = bee_df['date_time'].dt.month
bee_df["day"] = bee_df['date_time'].dt.day
bee_df["hour"] = bee_df['date_time'].dt.hour
bee_df["minute"] = bee_df['date_time'].dt.minute

bee_df["code"] = bee_df['location'].map(lambda x: x.split(',', 2)[1])

if st.checkbox('Show data'):
    st.subheader('Data')
    st.write(bee_df)


st.markdown(
    """
    ### And after that?
    By leveraging machine learning and other advanced technologies, we can pave the way towards a more sustainable
    and resilient future for our bees and our food system.

    ### About myself
    I am an AI engineer located in Toulouse. You can reach me out here:
"""
)

# Create a button that links to your LinkedIn profile
if st.button('LinkedIn'):
    linkedin_url = 'https://www.linkedin.com/in/tsara-ralantoson/'
    st.markdown(linkedin_url, unsafe_allow_html=True)

# Create a button that links to your GitHub profile
if st.button('GitHub'):
    github_url = 'https://github.com/tsaratiana'
    st.markdown(github_url, unsafe_allow_html=True)