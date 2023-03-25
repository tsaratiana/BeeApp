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


st.set_page_config(page_title="Health", page_icon="🧬")

st.markdown("# 🧬 Health")
st.write(
    """This page explores the health of bees."""
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


# # Group data by health
# health_count = bee_df.groupby("health").size().reset_index(name="count")

# # Create a bar chart using Plotly
# fig = px.bar(health_count, x="health", y="count")

# # Set plotly configuration options to rotate x-axis labels by 45 degrees
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.plotly_chart(fig, config={"xaxis": {"tickangle": 45}})

st.write("""Let us see the number of bees images for each health issue.
""")

selected_data = pse.plotly_events(
        show_image_counts("health"),
        click_event=True
    )

# Extract the selected data from the captured data
try:
    selected_category_value = selected_data[0]['x']
except IndexError:
    selected_category_value = None

# Display the images for the selected category value
if selected_category_value:
    draw_category_images("health", category=selected_category_value, cols=5)
else:
    st.write("**Please select a value for the category.**")

    


st.write("""Let us see the number of images per subspecies and health.
""")

pivot_df = bee_df.pivot_table(index='subspecies', columns='health', values='file', aggfunc='count')

# Create the heatmap using seaborn
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(pivot_df, cmap='YlOrBr', annot=True, fmt='.0f', ax=ax, annot_kws={'fontsize':8})

# Set the plot title and axis labels
# ax.set_title('Number of images per subspecies and health', fontsize=16)
ax.set_xlabel('health', fontsize=8, color='white')
ax.set_ylabel('subspecies', fontsize=8, color='white')

# Rotate the x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Modify the font size and color of the labels
ax.tick_params(axis='both', which='major', labelsize=7, labelcolor='white', colors='white')

# Modify the color of the heatmap
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8, labelcolor='white')

# Modify the background color to match the Streamlit theme
ax.set_facecolor('none')
fig = ax.get_figure()
fig.patch.set_alpha(0.0)

# Show the plot in Streamlit
st.pyplot(fig)




st.write("""Let us see the number of images per location and health, grouped by subspecies
""")

# Number of Images per Location and Health, Grouped by Subspecies
tmp = bee_df.groupby(['health', 'location'])['subspecies'].value_counts()
df = pd.DataFrame(data={'Images': tmp.values}, index=tmp.index).reset_index()

fig, ax = plt.subplots(figsize=(12, 8))

fig = px.scatter(
    df,
    x="location",
    y="health",
    size="Images",
    color="subspecies",
    hover_name="health"
)

st.plotly_chart(fig, use_container_width=False)


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")