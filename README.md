# Sound-and-Frequency-Visualization

Emotions are a vital part of communication in our daily lives. It helps us let others know how we feel about different things, what we want, etc. We want to know how we can recognize the difference between emotions and see if we can find any patterns.

The audience for this interactive visualization would be academic researchers or students to help them improve language learning or speech therapy. Teachers in grade school and in foreign language may also find this helpful to help individuals with speech disorders or non-native speakers who struggle with emotional nuances of a new language. Actors would also benefit from this to improve how they convey emotion on the big screen. 

The audio emotion dataset from [Kaggle](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions) contains audio for the same set of sentences (~2000 each) under different moods:
Angry, Happy, Sad, Neutral, Fearful, Disgusted, Surprised

The code utilizes libraries such as Dash, Plotly, NumPy, librosa, and FastDTW, and sets up the initial directory structure for storing audio files categorized by emotions. The layout consists of dropdowns for selecting sentences and emotions and various graphs to display spectrograms, time-domain waveforms, and pitch changes.

1. Spectrogram Analysis:
  - Original spectrograms of selected audio files are displayed, showing the frequency content over time for each emotion.
  - Aligned DTW spectrograms reveal how one audio file can be temporally warped to match the frequency content of another, highlighting similarities and differences.
  
2. Waveform Comparison:
  - Time-domain waveforms of the selected audio files are compared, showing the overall amplitude variations and differences in the audio signals.

3. Pitch Changes:
  - Pitch data extracted from the audio files is visualized, illustrating how the pitch varies over time for each emotion.
  - The log-transformed pitch data provides a clearer view of changes, aiding in the identification of emotional nuances.
