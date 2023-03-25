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

st.set_page_config(page_title="Exploration", page_icon="🔍")

st.markdown("# 🔍 Exploration")
st.write(
    """This page make a general data exploration of the dataset."""
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


# Define a function to display a bar chart of the number of images per subspecies
def show_image_counts(selected_category):
    # Group the data by subspecies and count the number of images
    subspecies_count = bee_df.groupby(selected_category).size().reset_index(name='count')

    # Create a histogram using Plotly
    fig = px.bar(subspecies_count, x=selected_category, y='count')
    fig.update_layout(
        clickmode='event+select',
        xaxis_title=selected_category,
        xaxis_title_standoff=15,
        yaxis_title_standoff=20,
        yaxis=dict(
                tickfont=dict(size=11, color='white'),
                gridcolor='rgba(255, 255, 255, 0.2)',
            ),
        yaxis_title="number of images", 
        margin=dict(l=50, r=100, t=30, b=110),
        font=dict(size=11, color='white'),
        font_family="'Nunito Sans', 'Helvetica Neue', Helvetica, sans-serif'",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)"
    )
    fig.update_traces(marker_color='#ebbd34')

    return fig

# Define a function to display a grid of images for a given category and subcategory
def draw_category_images(var, category=None, cols=5):
    if category:
        data = bee_df[bee_df[var] == category]
    else:
        data = bee_df

    categories = data[var].unique()

    if category and category not in categories:
        return st.write(f"No data found for {category}")

    images = data.sample(min(cols, len(data)))['file'].tolist()

    # Create a row of beta columns to display the images side by side
    columns = st.columns(cols)

    for i, image in enumerate(images):
        img = Image.open(IMAGE_PATH + image)
        img = img.resize((300, 300))
        with columns[i % 5]:
            st.image(img, use_column_width=True)



# Allow the user to select a category from a dropdown
category_options = ['subspecies', 'location', 'day', 'hour']
selected_category = st.selectbox('Select a category:', category_options)

if selected_category:
    # Display the bar chart of image counts and capture the selected data with streamlit-plotly-events
    selected_data = pse.plotly_events(
        show_image_counts(selected_category),
        click_event=True
    )

    # Extract the selected data from the captured data
    try:
        selected_category_value = selected_data[0]['x']
    except IndexError:
        selected_category_value = None

    # Display the images for the selected category value
    if selected_category_value:
        draw_category_images(selected_category, category=selected_category_value, cols=5)
    else:
        st.write("**Please select a value for the category.**")


st.write("""Let us see the number of bees images per date, approx. hour and location.
""")

# Number of bees images per date, approx. hour and location
tmp = bee_df.groupby(['date_time', 'hour'])['location'].value_counts()
df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

fig = px.scatter(
    df,
    x="date_time",
    y="hour",
    size="Images",
    color="location",
    hover_name="location"
)

st.plotly_chart(fig, use_container_width=False)


st.write("""Let us group the data by pollen carrying status.
""")


# Group the data by pollen carrying status
pollen_df = bee_df.groupby(['pollen_carrying']).size().reset_index(name='count')

# Create the donut chart using plotly
colors = ['#8e9e3c', '#8f3400']
fig = go.Figure(data=[go.Pie(labels=pollen_df['pollen_carrying'], values=pollen_df['count'], hole=.3)])
fig.update_layout(title='Pollen carrying distribution')
fig.update_traces(marker=dict(colors=colors))
# Show the chart in Streamlit
st.plotly_chart(fig)


st.write("""Only 0.348% bees are carrying pollen. To which species do they belong?
""")

# Create a list of subspecies to display in the radio button
subspecies_list = bee_df['subspecies'].unique().tolist()

# Add a radio button widget to select a subspecies
selected_subspecies = st.radio("Select a subspecies to view its pollen carrying distribution:", subspecies_list)



selected_bee_df = bee_df[bee_df['subspecies'] == selected_subspecies]
fig = px.histogram(selected_bee_df, x='pollen_carrying', barmode='group', nbins=10, color_discrete_sequence=['#FF5733'])
fig.update_layout(title=f"Pollen carrying distribution of {selected_subspecies}", xaxis_title='Pollen carrying', yaxis_title='Count')
st.plotly_chart(fig)


st.write("""Now, let us see the caste distribution.
""")
# Group the data by caste
caste_df = bee_df.groupby(['caste']).size().reset_index(name='count')

# Create the donut chart using plotly
fig = go.Figure(data=[go.Pie(labels=caste_df['caste'], values=caste_df['count'], hole=.3)])
fig.update_layout(title='Caste distribution')
fig.update_traces(marker=dict(colors=['#e38ab5']))
# Show the chart in Streamlit
st.plotly_chart(fig)

st.write("""All bees are workers! The queen is resting safely...
""")
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")