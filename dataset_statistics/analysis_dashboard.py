import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sympy import false


# Function to load and merge CSV files
def load_data(files):
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


# Function to display summary statistics and plots
def display_dashboard(dataset_df, dataset_name):
    # Filter dataset by selected dataset
    #dataset_df = df[df['dataset'] == dataset_name]

    # Number of videos, tracks, and tracks with at least one occlusion
    num_videos = dataset_df['video'].nunique()
    num_tracks = dataset_df['track'].shape[0]

    # group by video, take the max of video length (it's the same for all tracks in a video) and then sum
    #num_images = dataset_df['video_length'].sum() old version
    num_images = dataset_df.groupby('video')['video_length'].max().sum()


    num_tracks_with_occlusion = dataset_df[dataset_df['number_of_occlusions'] > 0]['track'].shape[0]

    # Create a row with three number stats
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        with st.container(border=True):
            st.metric("Number of videos", num_videos)

    with col2:
        with st.container(border=True):
            st.metric("Number of Images", num_images)

    with col3:
        with st.container(border=True):
            st.metric("Number of Tracks", num_tracks)

    with col4:
        with st.container(border=True):
            st.metric("Tracks with Occlusion", num_tracks_with_occlusion)

    with col5:
        with st.container(border=True):
            st.metric("Average track length", f'{np.mean(dataset_df["track_length"]):.2f}')

    with col6:
        with st.container(border=True):
            st.metric("Average Tracks per video", f'{num_tracks / num_videos:.2f}')

    col1, col2,_ = st.columns(3)

    with col1:
        # Track length histogram
        st.subheader("Video Length Distribution")
        with st.container(border=True):

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(dataset_df['video_length'], bins=10, kde=True, ax=ax, color="cornflowerblue")
            # remove border
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # draw line for average video length
            avg_seq_len = np.mean(dataset_df['video_length'])
            ax.axvline(avg_seq_len, color='goldenrod', linestyle='dashed', linewidth=1)
            ax.text(avg_seq_len, 1, f'{avg_seq_len:.2f}', color='goldenrod', ha='center', va='bottom', transform=ax.get_xaxis_transform())

            ax.set_title("Video Length Distribution\n", fontsize=16)
            st.pyplot(fig)

    with col2:
        # Track length histogram
        st.subheader("Track Length Distribution")
        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(dataset_df['track_length'], bins=10, kde=True, ax=ax)
            # remove border
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # draw line for average track length
            avg_track_len = np.mean(dataset_df['track_length'])
            ax.axvline(avg_track_len, color='goldenrod', linestyle='dashed', linewidth=1)
            ax.text(avg_track_len, 1, f'{avg_track_len:.2f}', color='goldenrod', ha='center', va='bottom', transform=ax.get_xaxis_transform())
            ax.set_title("Track Length Distribution\n", fontsize=16)
            st.pyplot(fig)


    col1, col2, col3 = st.columns(3)


    with col1:
        # Occlusion histogram
        st.subheader("Occlusion Distribution")

        subCol1, subCol2 = st.columns(2)
        with subCol1:
            occlusion_column = st.selectbox("Select Occlusion Metric",
                                        ['number_of_occlusions', 'avg_occlusion_length', 'min_occlusion_length',
                                         'max_occlusion_length'])
        with subCol2:
            st.container(height=9, border=False)
            exclude_no_occlusion = st.checkbox("Exclude rows with no occlusion")

        if exclude_no_occlusion:
            filtered_df = dataset_df[dataset_df[occlusion_column] > 0]
        else:
            filtered_df = dataset_df

        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_df[occlusion_column], bins=10, kde=True, ax=ax, color="forestgreen")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # draw line for average occlusion
            avg_occlusion = np.mean(filtered_df[occlusion_column])
            ax.axvline(avg_occlusion, color='goldenrod', linestyle='dashed', linewidth=1)
            ax.text(avg_occlusion, 1, f'{avg_occlusion:.2f}', color='goldenrod', ha='center', va='bottom', transform=ax.get_xaxis_transform())
            ax.set_title(f"{occlusion_column} Distribution\n", fontsize=16)
            st.pyplot(fig)


    with col2:
        # Scale histogram
        st.subheader("Scale Distribution")

        subCol1, subCol2 = st.columns(2)
        with subCol1:
            scale_column = st.selectbox("Select Scale Metric", ['max_scale', 'avg_scale', 'min_scale', 'number_of_scale_changes', 'avg_scale_change'])
        with subCol2:
            st.container(height=9, border=False)
            exclude_no_scale = false

        if exclude_no_scale:
            filtered_df = dataset_df[dataset_df[scale_column].notna()]
        else:
            filtered_df = dataset_df

        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_df[scale_column], bins=10, kde=True, ax=ax, color="gold")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # draw line for average scale
            avg_scale = np.mean(filtered_df[scale_column])
            ax.axvline(avg_scale, color='goldenrod', linestyle='dashed', linewidth=1)
            ax.text(avg_scale, 1, f'{avg_scale:.2f}', color='goldenrod', ha='center', va='bottom', transform=ax.get_xaxis_transform())
            ax.set_title(f"{scale_column} Distribution\n", fontsize=16)
            st.pyplot(fig)


    with col3:
        # Speed histogram
        st.subheader("Speed Distribution")
        speed_column = 'avg_speed'  # only available column for speed

        subCol1, subCol2 = st.columns(2)
        with subCol1:
            speed_column = st.selectbox("Select Scale Metric",
                                        ['avg_speed', 'number_of_abrupt_motion_changes'])
        with subCol2:
            pass


        with st.container(border=True):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_df[speed_column], bins=10, kde=True, ax=ax, color="violet")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # draw line for average speed
            avg_speed = np.mean(filtered_df[speed_column])
            ax.axvline(avg_speed, color='goldenrod', linestyle='dashed', linewidth=1)
            ax.text(avg_speed, 1, f'{avg_speed:.2f}', color='goldenrod', ha='center', va='bottom', transform=ax.get_xaxis_transform())
            ax.set_title(f"{speed_column} Distribution\n", fontsize=16)
            st.pyplot(fig)



    # Display the dataframe
    st.subheader("Dataset Overview")
    st.write(dataset_df)


# Main Streamlit app
def main():
    st.set_page_config(layout="wide")  # Set the layout to wide

    st.title('Dataset Analysis Dashboard')

    # Upload a specific file
    file_name = [
        'statistics_GOT.csv',
        'statistics_LASOT.csv',
        'statistics_TAO.csv',
        'statistics_BDD.csv',
        'statistics_OVTB.csv',
        'statistics_ANIMALTRACK.csv',
        'statistics_BFT.csv',
        'statistics_DANCETRACK.csv',
        'statistics_SPORTSMOT.csv',

        'statistics_GOT_TRAIN.csv',
        'statistics_LASOT_TRAIN.csv',
        'statistics_BDD_TRAIN.csv',
        'statistics_ANIMALTRACK_TRAIN.csv',
        'statistics_BFT_TRAIN.csv',
        'statistics_DANCETRACK_TRAIN.csv',
        'statistics_SPORTSMOT_TRAIN.csv',
    ]

    # Load and merge data
    df = load_data(file_name)

    # Display dataset names in a dropdown menu
    dataset_names = df['dataset'].unique()

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        selected_dataset = st.selectbox("Select Dataset", ["All Datasets"] + list(dataset_names))

    with col2:
        min_track_length = st.number_input("Min Track Length", min_value=0, value=0)

    with col3:
        min_occlusions = st.number_input("Min Occlusions", min_value=0, value=0)

    with col4:
        min_video_length = st.number_input("Min Video Length", min_value=0, value=0)

    with col5:
        min_max_occ_length = st.number_input("Min Max Occ Length", min_value=0, value=0)

    # Apply filters
    filtered_df = df[
        ((df['dataset'] == selected_dataset) | (selected_dataset == "All Datasets")) &
        (df['track_length'] >= min_track_length) &
        ((df['number_of_occlusions'] >= min_occlusions) | (df['max_occlusion_length'] >= min_max_occ_length)) &
        (df['video_length'] >= min_video_length)
        ]



    # Display dashboard for the selected dataset
    display_dashboard(filtered_df, selected_dataset)


if __name__ == "__main__":
    main()
