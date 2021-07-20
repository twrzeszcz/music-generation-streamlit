import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import gc
from fractions import Fraction
import seaborn as sns
import librosa
import librosa.display
import pandas as pd
from music21 import *
import io
import pretty_midi
from scipy.io import wavfile
import matplotlib.pyplot as plt
import cv2

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_token_model():
    model_combined = keras.models.load_model('models/combined_full_gen.h5')
    return model_combined

@st.cache(allow_output_mutation=True)
def load_separate_model():
    notes_gen = keras.models.load_model('models/notes_gen.h5')
    durations_gen = keras.models.load_model('models/durations_gen.h5')
    offsets_gen = keras.models.load_model('models/offsets_gen.h5')
    return (notes_gen, durations_gen, offsets_gen)

@st.cache(allow_output_mutation=True)
def load_model_full():
    model_full = keras.models.load_model('models/model_full.h5')
    return model_full

@st.cache
def read_utils_tokens():
    num_to_tokens = pickle.load(open('utils/num_to_tokes.pkl', 'rb'))
    unique_tokens = pickle.load(open('utils/unique_tokens.pkl', 'rb'))
    return num_to_tokens, unique_tokens

@st.cache
def read_utils_separate():
    num_to_notes = pickle.load(open('utils/num_to_notes.pkl', 'rb'))
    num_to_dur = pickle.load(open('utils/num_to_dur.pkl', 'rb'))
    num_to_off = pickle.load(open('utils/num_to_off.pkl', 'rb'))
    unique_notes = pickle.load(open('utils/unique_notes.pkl', 'rb'))
    unique_durations = pickle.load(open('utils/unique_durations.pkl', 'rb'))
    unique_offsets = pickle.load(open('utils/unique_offsets.pkl', 'rb'))
    return num_to_notes, num_to_dur, num_to_off, unique_notes, unique_durations, unique_offsets

def gen_pred_tokens(n_starting_points, n_steps, temperature_token, model):
    num_to_tokens, unique_tokens = read_utils_tokens()
    sample_tokens = np.random.randint(0, len(unique_tokens), (1, n_starting_points, 1))
    for step in range(n_steps):
        logits_tokens = model.predict(sample_tokens)[:, -1, :]
        next_token = np.array(tf.random.categorical(tf.math.log(logits_tokens) / temperature_token, num_samples=1))[
            0].reshape(1, 1, 1)
        sample_tokens = np.concatenate([sample_tokens, next_token], axis=1)

    sample_tokens_mapped = list(map(num_to_tokens.get, list(sample_tokens.reshape(-1))))

    return sample_tokens_mapped


def gen_pred_separate(n_starting_points, n_steps, temperature_note, temperature_duration, temperature_offset, models):
    model_notes, model_durations, model_offsets = models
    num_to_notes, num_to_dur, num_to_off, unique_notes, unique_durations, unique_offsets = read_utils_separate()
    sample_notes = np.random.randint(0, len(unique_notes), (1, n_starting_points, 1))
    sample_dur = np.random.randint(0, len(unique_durations), (1, n_starting_points, 1))
    sample_off = np.random.randint(0, len(unique_offsets), (1, n_starting_points, 1))

    for step in range(n_steps):
        logits_notes = model_notes.predict(sample_notes)[:, -1, :]
        logits_dur = model_durations.predict(sample_dur)[:, -1, :]
        logits_off = model_offsets.predict(sample_off)[:, -1, :]

        next_note = np.array(tf.random.categorical(tf.math.log(logits_notes) / temperature_note, num_samples=1))[
            0].reshape(1, 1, 1)
        next_dur = np.array(tf.random.categorical(tf.math.log(logits_dur) / temperature_duration, num_samples=1))[0].reshape(
            1, 1, 1)
        next_off = np.array(tf.random.categorical(tf.math.log(logits_off) / temperature_offset, num_samples=1))[0].reshape(
            1, 1, 1)

        sample_notes = np.concatenate([sample_notes, next_note], axis=1)
        sample_dur = np.concatenate([sample_dur, next_dur], axis=1)
        sample_off = np.concatenate([sample_off, next_off], axis=1)

    sample_notes_mapped = list(map(num_to_notes.get, list(sample_notes.reshape(-1))))
    sample_dur_mapped = list(map(num_to_dur.get, list(sample_dur.reshape(-1))))
    sample_off_mapped = list(map(num_to_off.get, list(sample_off.reshape(-1))))

    return sample_notes_mapped, sample_dur_mapped, sample_off_mapped


def gen_pred_full(n_starting_points, n_steps, temperature_note, temperature_duration, temperature_offset, model):
    num_to_notes, num_to_dur, num_to_off, unique_notes, unique_durations, unique_offsets = read_utils_separate()
    sample_notes = tf.convert_to_tensor(np.random.randint(0, len(unique_notes), (1, n_starting_points)))
    sample_dur = tf.convert_to_tensor(np.random.randint(0, len(unique_durations), (1, n_starting_points)))
    sample_off = tf.convert_to_tensor(np.random.randint(0, len(unique_offsets), (1, n_starting_points)))
    gen_pred = [sample_notes, sample_dur, sample_off]


    for step in range(n_steps):
        pred = model.predict(gen_pred)

        next_note = tf.random.categorical(tf.math.log(pred[0][:, -1, :]) / temperature_note, num_samples=1)
        next_dur = tf.random.categorical(tf.math.log(pred[1][:, -1, :]) / temperature_duration, num_samples=1)
        next_off = tf.random.categorical(tf.math.log(pred[2][:, -1, :]) / temperature_offset, num_samples=1)

        gen_pred = [np.concatenate([gen_pred[i], j], axis=1) for i, j in enumerate([next_note, next_dur, next_off])]

    sample_notes_mapped = list(map(num_to_notes.get, list(gen_pred[0].flatten())))
    sample_dur_mapped = list(map(num_to_dur.get, list(gen_pred[1].flatten())))
    sample_off_mapped = list(map(num_to_off.get, list(gen_pred[2].flatten())))

    return sample_notes_mapped, sample_dur_mapped, sample_off_mapped


def gen_midi_tokens(pred):
    offset = Fraction(0.0)
    output_notes = []
    for pattern in pred:
        pattern_split = pattern.split('_')
        if ('.' in pattern_split[0]) or pattern_split[0].isdigit():
            offset += Fraction(pattern_split[2])
            notes_in_chord = pattern_split[0].split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = duration.Duration(Fraction(pattern_split[1]))
            output_notes.append(new_chord)
        elif pattern_split[0] == 'rest':
            offset += Fraction(pattern_split[2])
            new_note = note.Rest(pattern_split[0])
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = duration.Duration(Fraction(pattern_split[1]))
            output_notes.append(new_note)
        else:
            offset += Fraction(pattern_split[2])
            new_note = note.Note(pattern_split[0])
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = duration.Duration(Fraction(pattern_split[1]))
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    filepath = midi_stream.write('midi', fp='temp.mid')

    return filepath

def gen_midi_separate(pred):
    pred_notes, pred_dur, pred_off = pred
    offset = Fraction(0.0)
    output_notes = []
    for pattern, dur, off in zip(pred_notes, pred_dur, pred_off):
        if ('.' in pattern) or pattern.isdigit():
            offset += Fraction(off)
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = duration.Duration(dur)
            output_notes.append(new_chord)
        elif pattern == 'rest':
            offset += Fraction(off)
            new_note = note.Rest(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = duration.Duration(dur)
            output_notes.append(new_note)
        else:
            offset += Fraction(off)
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = duration.Duration(dur)
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    filepath = midi_stream.write('midi', fp='temp.mid')

    return filepath

def main_section_generation():
    st.title('Classical Music Generation')
    st.markdown('This is a music generation project where we try to generate classical music with an RNN. Model was trained on the parsed **MIDI** '
                'files and generated pieces are also saved in this format. One can choose from 3 different models. Frist model named **Tokens** '
                'generates tokens which are combinations of a note, duration and an offset. Second model named **Separate models** consists of '
                '3 trained models to predict notes, durations and offsets separately. The third model named **Single model separate outputs** '
                'is a single model which has 3 inputs and 3 outputs. Generation is done by selecting number of random '
                'starting points. Then the user can choose how many points should be generated. Points are generated according to their '
                'predicted probabilities so not the most probable note is choosen as the next one. Temperature controls relative '
                'probabilities. If it is *1.0* then probabilities are the same as given by the model. When it is increased above *1.0*, then '
                'probabilities are more equalized and this introduces more randomness. If it is below *1.0* then only most likely notes will be chosen. ')

    model_type = st.sidebar.selectbox('Select type of the model', ['Tokens', 'Separate models', 'Single model separate outputs'])
    if model_type == 'Tokens':
        model_combined = load_token_model()
        n_starting_points = st.sidebar.slider('Choose number of random starting points', 1, 20)
        n_steps = st.sidebar.slider('Choose number of generated steps', 1, 200)
        temperature_token = st.sidebar.slider('Choose temperature', 0.1, 2.0, 1.0)
        if st.sidebar.button('Generate music'):
            pred = gen_pred_tokens(n_starting_points, n_steps, temperature_token, model=model_combined)
            filepath = gen_midi_tokens(pred)

            with st.spinner(f"Transcribing to FluidSynth"):
                midi_data = pretty_midi.PrettyMIDI(filepath)
                audio_data = midi_data.fluidsynth()
                audio_data = np.int16(
                    audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
                )
                virtualfile = io.BytesIO()
                wavfile.write(virtualfile, 44100, audio_data)
            st.audio(virtualfile)

            st.success('Successfully generated')
            del model_combined
            gc.collect()

    if model_type == 'Separate models':
        models = load_separate_model()
        n_starting_points = st.sidebar.slider('Choose number of random starting points', 1, 20)
        n_steps = st.sidebar.slider('Choose number of generated steps', 1, 200)
        temperature_note = st.sidebar.slider('Choose temperature note', 0.1, 2.0, 1.0)
        temperature_duration = st.sidebar.slider('Choose temperature duration', 0.1, 2.0, 1.0)
        temperature_offset = st.sidebar.slider('Choose temperature offset', 0.1, 2.0, 1.0)
        if st.sidebar.button('Generate music'):
            pred = gen_pred_separate(n_starting_points, n_steps, temperature_note, temperature_duration, temperature_offset, models=models)
            filepath = gen_midi_separate(pred)

            with st.spinner(f"Transcribing to FluidSynth"):
                midi_data = pretty_midi.PrettyMIDI(filepath)
                audio_data = midi_data.fluidsynth()
                audio_data = np.int16(
                    audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
                )
                virtualfile = io.BytesIO()
                wavfile.write(virtualfile, 44100, audio_data)
            st.audio(virtualfile)

            st.success('Successfully generated')
            del models
            gc.collect()

    if model_type == 'Single model separate outputs':
        model = load_model_full()
        n_starting_points = st.sidebar.slider('Choose number of random starting points', 1, 20)
        n_steps = st.sidebar.slider('Choose number of generated steps', 1, 200)
        temperature_note = st.sidebar.slider('Choose temperature note', 0.1, 2.0, 1.0)
        temperature_duration = st.sidebar.slider('Choose temperature duration', 0.1, 2.0, 1.0)
        temperature_offset = st.sidebar.slider('Choose temperature offset', 0.1, 2.0, 1.0)
        if st.sidebar.button('Generate music'):
            pred = gen_pred_full(n_starting_points, n_steps, temperature_note, temperature_duration, temperature_offset, model=model)
            filepath = gen_midi_separate(pred)

            with st.spinner(f"Transcribing to FluidSynth"):
                midi_data = pretty_midi.PrettyMIDI(filepath)
                audio_data = midi_data.fluidsynth()
                audio_data = np.int16(
                    audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
                )
                virtualfile = io.BytesIO()
                wavfile.write(virtualfile, 44100, audio_data)
            st.audio(virtualfile)

            st.success('Successfully generated')
            del model
            gc.collect()




@st.cache(allow_output_mutation=True)
def load_model_classification():
    model = keras.models.load_model('models/music_classifier_efficientnet_trimmed.h5')
    return model

def main_section_classification():
    st.title('Music Genre Classification')
    st.markdown(
        'This is a simple music genre classification app where one can upload a *wav* audio file and get the predicted genre. '
        'Genres are predicted based on the spectrograms generated from 3s audio clips. Therefore the uploaded audio track has to be also '
        'trimmed to 3 seconds. User has to input an offset in seconds which indicates the starting point of the 3 seconds interval. '
        'Additionally one can also display the spectrogram for the entire audio file.')

    audio_file = st.sidebar.file_uploader('Choose wav file to upload', type=['wav'])
    if audio_file is not None:
        if st.sidebar.button('Display spectrogram'):
            y, sr = librosa.load(audio_file)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(15, 5))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
            ax.axis('off')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)

        if st.sidebar.checkbox('Prediction'):
            starting_point = st.text_input('Specify the offset')
            if st.sidebar.button('Predict genre'):
                y, sr = librosa.load(audio_file, offset=float(starting_point), duration=3)
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_dB = librosa.power_to_db(S, ref=np.max)

                # plot to numpy
                fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                ax.axis('off')
                fig.tight_layout(pad=0)
                io_buf = io.BytesIO()
                fig.savefig(io_buf, format='raw', dpi=100)
                io_buf.seek(0)
                img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
                io_buf.close()

                class_mapping = {0: 'blues',
                                 1: 'classical',
                                 2: 'country',
                                 3: 'disco',
                                 4: 'hiphop',
                                 5: 'jazz',
                                 6: 'metal',
                                 7: 'pop',
                                 8: 'reggae',
                                 9: 'rock'}
                model = load_model_classification()
                img_prep = np.expand_dims(cv2.resize(img_arr, (224, 224))[:, :, :3], axis=0)
                y_pred = model.predict(img_prep)
                df = pd.DataFrame(y_pred.flatten(), index=class_mapping.values(),
                                  columns=['Probabilities']).sort_values(
                    by='Probabilities', ascending=False)
                st.dataframe(df)

                fig, ax = plt.subplots()
                sns.barplot(data=df, y=df.index, x=df.Probabilities)
                st.pyplot(fig)


options = ['Main', 'Music generation', 'Music genre classification']
option = st.sidebar.selectbox('Select module', options)

if option == 'Main':
    st.title('Music generation and classification app')
    st.image(
        'https://images.pexels.com/photos/594388/vinyl-record-player-retro-594388.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',
        use_column_width=True)
    st.markdown('This is a machine learning project for music generation and classification. There are 2 available modules: *Music generation* section '
                'is used to generate classical music and in the *Music genre classification* section a user can predict the music genre '
                'on the uploaded audio file. Trained models and notebooks used for preprocessing and training can be found here '
                '[GitHub](https://github.com/twrzeszcz/music-generation-streamlit) and here '
                '[GitHub](https://github.com/twrzeszcz/music-genre-classification).')

if option == 'Music genre classification':
    main_section_classification()
    gc.collect()

if option == 'Music generation':
    main_section_generation()
    gc.collect()
