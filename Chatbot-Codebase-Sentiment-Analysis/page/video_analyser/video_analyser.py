import streamlit as st
from moviepy.editor import VideoFileClip
import tempfile
from datasets import load_dataset
from PIL import Image
from pydub import AudioSegment
import os
from models import model
import shutil
import time

pipe = model.create_speech_recognition_pipeline()

def split_audio(file_path, segment_length=15):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Calculate the number of segments
    segment_length_ms = segment_length * 1000  # convert to milliseconds
    num_segments = len(audio) // segment_length_ms + 1
    
    # Create output directory
    output_dir = "temp_files/audio/output_segments"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = start_time + segment_length_ms
        
        # Extract the segment
        segment = audio[start_time:end_time]
        
        # Export the segment
        segment.export(os.path.join(output_dir, f"segment_{i + 1}.mp3"), format="mp3")

def delete_all_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through all items in the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # Check if it's a file or a directory and remove it accordingly
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"All contents deleted from: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")

def extract_segment_number(file):
    return int(file['path'].split('segment_')[1].split('.mp3')[0]) 
 
@st.cache_data(show_spinner=False)
def transcribe_audio(audio_file_path):
    # model = whisper.load_model("base")
    # result = model.transcribe(audio_file_path)
    delete_all_contents("temp_files")
    os.makedirs("temp_files/audio/output_segments", exist_ok=True)
    split_audio(audio_file_path)
    dataset = load_dataset("temp_files/audio/output_segments")
    audio_files = [audio['audio'] for audio in dataset['train']]
    sorted_audio_files = sorted(audio_files, key=extract_segment_number)
    result = pipe(sorted_audio_files)
    # Format the transcription with timestamps
    transcription_with_timestamps = []
    idx = 0
    for res in result:
        total_seconds = idx * 15
        formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_seconds))
        transcription_with_timestamps.append((total_seconds, f'<a href="#{formatted_time}" style="color: blue;">{formatted_time}</a> - {res['text']}'))
        idx+=1
     
    return transcription_with_timestamps
 
def format_time(seconds):
    """Helper function to format time in HH:MM:SS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
 
# Generate thumbnail from video
def generate_thumbnail(video_path, thumbnail_path, time=1):
    clip = VideoFileClip(video_path)
    thumbnail = clip.get_frame(time)
    thumbnail_image = Image.fromarray(thumbnail)
    thumbnail_image.save(thumbnail_path)
    return thumbnail_path
 
def main(domain_name):
    st.markdown("""
    <div style='text-align: left; margin-left: 60px; margin-bottom: 0px'>
        <h3> Video Captioning and Translating🤖</h3>
    </div>
    """, unsafe_allow_html=True)
 
    # File uploader for multiple video files
    video_files = st.file_uploader("Upload video files", type=["mp4", "avi", "mov"], accept_multiple_files=True)
 
    if video_files:
        # Create a grid layout for thumbnails
        num_videos = len(video_files)
        cols = st.columns(4)  # Adjust the number of columns for the grid layout
 
        if 'transcriptions' not in st.session_state:
            st.session_state.transcriptions = {}
 
        temp_dir = tempfile.mkdtemp()
        video_path = None  # Store the path of the currently playing video
 
        # Iterate over each uploaded video and display thumbnails
        for i, video_file in enumerate(video_files):
            col = cols[i % 4]  # Dynamically assign columns to videos
 
            # Save the uploaded video to a temporary location for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir) as temp_file:
                temp_file.write(video_file.read())
                video_path = temp_file.name
 
            # Generate a thumbnail
            thumbnail_path = os.path.join(temp_dir, f"{i}_thumbnail.png")
            generate_thumbnail(video_path, thumbnail_path)
 
            # Display the thumbnail in the grid
            with col:
                # Extract the first two words from the video file name
                video_file_name = video_file.name.split(".")[0]  # Remove file extension
                first_two_words = " ".join(video_file_name.split()[:1])  # Get the first two words
 
                # Display the expander with the first two words of the video file name
                with st.expander(f"Play {first_two_words}", expanded=True):
                    # Display the video in the expander
                    st.video(video_path)
 
                if st.button(f"Transcribe {first_two_words}", key=f"transcribe_{i}"):
                    # Display a message while the transcription is processing
                    with st.spinner("Transcribing the video..."):
                        # Extract audio from video using moviepy
                        clip = VideoFileClip(video_path)
                        audio_path = video_path.replace(".mp4", ".wav")
                        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                            clip.audio.write_audiofile(audio_path)
 
                        # Transcribe the audio with timestamps
                        transcription_segments = transcribe_audio(audio_path)
 
                        # Store transcription in session state
                        st.session_state.transcriptions[video_file.name] = transcription_segments
 
                    # Display transcription after processing is done
                    st.success("Transcription completed! Below is the transcription:")
 
                # Display transcription if already available
                if video_file.name in st.session_state.transcriptions:
                    transcription_segments = st.session_state.transcriptions[video_file.name]
 
                    # Display transcription for the current video in an expander
                    with st.expander(f"Transcription for {video_file.name}", expanded=False):
                        for start_time, transcription in transcription_segments:
                            # Create a button for each timestamp
                            if st.button(f"Jump to {format_time(start_time)}", key=f"jump_{i}_{start_time}"):
                                # Update session state with the start time
                                st.session_state.video_time = start_time
                            st.markdown(transcription, unsafe_allow_html=True)