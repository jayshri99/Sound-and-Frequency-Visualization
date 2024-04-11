import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from fastdtw import fastdtw
import pandas as pd
import os

app = dash.Dash(__name__)

# Assuming the files are structured under a main directory named 'emotions'
BASE_DIR = 'emotions'

def get_audio_path(emotion, sentence):
    return os.path.join(BASE_DIR, emotion, f"{sentence}.wav")

def compute_stft(audio_path):
    audio, sr = librosa.load(audio_path)
    return np.abs(librosa.stft(audio))

def apply_dtw(stft_audio1, stft_audio2):
    distance, path = fastdtw(stft_audio1.T, stft_audio2.T, dist=euclidean)
    path = np.array(path)
    path_audio2 = path[:, 1]
    warped_audio2_initial = stft_audio2[:, path_audio2]

    num_frames_audio1 = stft_audio1.shape[1]  # Number of time frames in audio1
    original_indices = np.linspace(0, len(warped_audio2_initial[0]) - 1, num=num_frames_audio1)

    # Initialize an empty matrix to hold the resampled audio2
    resampled_audio2 = np.zeros((stft_audio2.shape[0], num_frames_audio1))

    # Interpolate each frequency bin independently
    for i in range(stft_audio2.shape[0]):  # Iterate over each frequency bin
        # Create an interpolation function for the current frequency bin
        interp_func = interp1d(np.arange(len(warped_audio2_initial[i])), warped_audio2_initial[i], kind='linear')
        
        # Resample the current frequency bin using the new time frame indices
        resampled_audio2[i] = interp_func(original_indices)

    return resampled_audio2

def generate_spectrogram_figure(stft_data, title):
    db_stft = librosa.amplitude_to_db(stft_data, ref=np.max)
    fig = px.imshow(db_stft, aspect='auto', color_continuous_scale='viridis', origin='lower')
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Frequency', coloraxis_colorbar=dict(title='dB'))
    return fig

def generate_wave_figure(sound1_path, sound2_path, title):
    sound1, _ = librosa.load(sound1_path)
    sound2, _ = librosa.load(sound2_path)
    
    min_len = min(len(sound1), len(sound2))

    df2 = pd.DataFrame({
        'Sample Index': range(min_len),  # Assuming audio1 and audio2 are the same length
        'Audio 1': sound1[:min_len],
        'Audio 2': sound2[:min_len]
    })

    # Plot using plotly.express with named legends
    fig = px.line(df2, x='Sample Index', y='Audio 1', title=title)
    fig.data[0].name = 'Audio 1'  # Renaming the legend for 'Audio 1'
    
    fig.add_scatter(x=df2['Sample Index'], y=df2['Audio 2'], mode='lines', name='Audio 2')
    
    fig.update_traces(opacity=0.65, showlegend = True)
    fig.update_layout(legend_title_text='Legend')

    return fig

def get_pitch_data(audio_path, audio_label):
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Extract pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Select the dominant pitch for each frame
    dominant_pitches = np.array([pitches[mag.argmax(), idx] for idx, mag in enumerate(magnitudes.T)])
    
    # Frames to time
    frames = range(len(dominant_pitches))
    t = librosa.frames_to_time(frames, sr=sr)

    temp = np.log(dominant_pitches + 1e-6)
    temp[temp < 0] = np.min(temp[temp>0]) - 0.1
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Time': t,
        'Log_Pitch': temp,  # Log scale to make changes more perceptible
        'Audio': audio_label
    })
    return df


app.layout = html.Div([
    html.H1("Sound & Freq Visualization", style={'textAlign': 'center'}),
    html.H2("Select Two Emotions Below to Compare", style={'textAlign': 'left'}),
    html.Div([
        html.Label('Select a sentence: ', style={'display': 'inline-block', 'margin-right': '10px'}),
        dcc.Dropdown(
            id='sentence-dropdown',
            options=[
                {'label': 'Kids_are_talking_by_the_door', 'value': 'Kids_are_talking_by_the_door'},
                {'label': 'Dogs_are_sitting_by_the_door', 'value': 'Dogs_are_sitting_by_the_door'},
                {'label': 'Say_the_word_back', 'value': 'Say_the_word_back'},
                {'label': 'Say_the_word_bar', 'value': 'Say_the_word_bar'},
            ],
            value='Kids_are_talking_by_the_door'
        ),
    ], style={'margin-bottom': '10px'}),
    html.Div([
        html.Label('Select a first emotion: ', style={'display': 'inline-block', 'margin-right': '10px'}),
        dcc.Dropdown(
            id='emotion-1-dropdown',
            options=[
                {'label': 'Angry', 'value': 'Angry'},
                {'label': 'Disgusted', 'value': 'Disgusted'},
                {'label': 'Fearful', 'value': 'Fearful'},
                {'label': 'Happy', 'value': 'Happy'},
                {'label': 'Neutral', 'value': 'Neutral'},
                {'label': 'Sad', 'value': 'Sad'},
                {'label': 'Surprised', 'value': 'Surprised'},
            ],
            value='Angry'
        ),
    ], style={'margin-bottom': '10px'}),
    html.Div([
        html.Label('Select a second emotion: ', style={'display': 'inline-block', 'margin-right': '10px'}),
        dcc.Dropdown(
            id='emotion-2-dropdown',
            options=[
                {'label': 'Angry', 'value': 'Angry'},
                {'label': 'Disgusted', 'value': 'Disgusted'},
                {'label': 'Fearful', 'value': 'Fearful'},
                {'label': 'Happy', 'value': 'Happy'},
                {'label': 'Neutral', 'value': 'Neutral'},
                {'label': 'Sad', 'value': 'Sad'},
                {'label': 'Surprised', 'value': 'Surprised'},
            ],
            value='Happy'
        ),
    ], style={'margin-bottom': '10px'}),

    html.Button('Play Emotion 1 Audio', id='play-emotion-1-btn', n_clicks=0),
    html.Button('Play Emotion 2 Audio', id='play-emotion-2-btn', n_clicks=0),
    html.Audio(id='emotion-1-audio', controls=True, style={'display': 'none'}),
    html.Audio(id='emotion-2-audio', controls=True, style={'display': 'none'}),

    html.H2("Original Spectrograms", style={'textAlign': 'left'}),
    html.Div([
        dcc.Graph(id='time_graph'),
        dcc.Graph(id='stft-before-dtw-emotion-1-top'),
        dcc.Graph(id='stft-before-dtw-emotion-2')
    ]),
    html.H2("Aligned DTW Spectrograms", style={'textAlign': 'left'}),
    html.Div([
        dcc.Graph(id='stft-before-dtw-emotion-1-bottom'),
        dcc.Graph(id='stft-after-dtw'),
        dcc.Graph(id='difference')
    ]),
    html.H2("Compare Pitch Changes", style={'textAlign': 'left'}),
    html.Div([
        dcc.Graph(id='pitch')
    ])
])

@app.callback(
    [Output('time_graph', 'figure'),
     Output('stft-before-dtw-emotion-1-top', 'figure'),
     Output('stft-before-dtw-emotion-2', 'figure')],
    [Input('emotion-1-dropdown', 'value'),
     Input('emotion-2-dropdown', 'value'),
     Input('sentence-dropdown', 'value')]
)
def update_figures1(selected_emotion_1, selected_emotion_2, selected_sentence):
    audio1_path = get_audio_path(selected_emotion_1, selected_sentence)
    audio2_path = get_audio_path(selected_emotion_2, selected_sentence)
    
    stft_audio1 = compute_stft(audio1_path)
    stft_audio2 = compute_stft(audio2_path)
    
    # Generate spectrograms
    fig_time = generate_wave_figure(audio1_path, audio2_path, f'Audios in time domain')
    fig_before_emotion_1 = generate_spectrogram_figure(stft_audio1, f'STFT Spectrum of {selected_sentence} - {selected_emotion_1} Emotion Before DTW')
    fig_before_emotion_2 = generate_spectrogram_figure(stft_audio2, f'STFT Spectrum of {selected_sentence} - {selected_emotion_2} Emotion Before DTW')
    return fig_time, fig_before_emotion_1, fig_before_emotion_2

@app.callback(
    [Output('stft-before-dtw-emotion-1-bottom', 'figure'),
     Output('stft-after-dtw', 'figure'),
     Output('difference', 'figure')],
    [Input('emotion-1-dropdown', 'value'),
     Input('emotion-2-dropdown', 'value'),
     Input('sentence-dropdown', 'value')]
)
def update_figures(selected_emotion_1, selected_emotion_2, selected_sentence):
    audio1_path = get_audio_path(selected_emotion_1, selected_sentence)
    audio2_path = get_audio_path(selected_emotion_2, selected_sentence)
    
    stft_audio1 = compute_stft(audio1_path)
    stft_audio2 = compute_stft(audio2_path)
    
    warped_audio2 = apply_dtw(stft_audio1, stft_audio2)
    
    # Generate spectrograms
    fig_before_emotion_1 = generate_spectrogram_figure(stft_audio1, f'STFT Spectrum of {selected_sentence} - {selected_emotion_1} Emotion Before DTW')
    fig_after_dtw = generate_spectrogram_figure(warped_audio2, f'STFT Spectrum of {selected_sentence} - {selected_emotion_2} Emotion After DTW')
    fig_diff = generate_spectrogram_figure(stft_audio1 - warped_audio2, f'STFT Spectrum of {selected_sentence} - {selected_emotion_2} Difference')
    
    return fig_before_emotion_1, fig_after_dtw, fig_diff

@app.callback(
    [Output('pitch', 'figure')],
    [Input('emotion-1-dropdown', 'value'),
     Input('emotion-2-dropdown', 'value'),
     Input('sentence-dropdown', 'value')]
)
def update_pitch(selected_emotion_1, selected_emotion_2, selected_sentence):
    audio1_path = get_audio_path(selected_emotion_1, selected_sentence)
    audio2_path = get_audio_path(selected_emotion_2, selected_sentence)

    # Load and prepare pitch data for both audio files
    df1 = get_pitch_data(audio1_path, 'Audio 1')
    df2 = get_pitch_data(audio2_path, 'Audio 2')
    # print(df1.columns)

    # Concatenate DataFrames
    df = pd.concat([df1, df2])
    # print(df.columns)

    # Plot using Plotly Express
    fig = px.line(df, x='Time', y='Log_Pitch', color='Audio', title='Pitch Changes Over Time')
    fig.update_yaxes(title_text='Log(Pitch)')
    fig.update_xaxes(title_text='Time (s)')

    return [fig]

# Callback to play Emotion 1 audio
@app.callback(
    Output('emotion-1-audio', 'src'),
    [Input('play-emotion-1-btn', 'n_clicks')],
    [State('emotion-1-dropdown', 'value'), State('sentence-dropdown', 'value')]
)
def play_emotion_1_audio(n_clicks, emotion, sentence):
    if n_clicks > 0:
        audio_path = get_audio_path(emotion, sentence)
        return app.get_asset_url(audio_path)
    raise PreventUpdate

# Callback to play Emotion 2 audio
@app.callback(
    Output('emotion-2-audio', 'src'),
    [Input('play-emotion-2-btn', 'n_clicks')],
    [State('emotion-2-dropdown', 'value'), State('sentence-dropdown', 'value')]
)
def play_emotion_2_audio(n_clicks, emotion, sentence):
    if n_clicks > 0:
        audio_path = get_audio_path(emotion, sentence)
        return app.get_asset_url(audio_path)
    raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=True)
