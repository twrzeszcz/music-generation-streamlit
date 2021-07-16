import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import gc
from fractions import Fraction
from music21 import *

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_token_model():
    model_combined = keras.models.load_model('model/combined_full_gen.h5')
    return model_combined

@st.cache(allow_output_mutation=True)
def load_separate_model():
    notes_gen = keras.models.load_model('model/notes_gen.h5')
    durations_gen = keras.models.load_model('model/durations_gen.h5')
    offsets_gen = keras.models.load_model('model/offsets_gen.h5')
    return (notes_gen, durations_gen, offsets_gen)

@st.cache(allow_output_mutation=True)
def load_model_full():
    model_full = keras.models.load_model('model/model_full.h5')
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


def gen_midi_tokens(pred, path_to_save):
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
    midi_stream.write('midi', fp=path_to_save + '.mid')

def gen_midi_separate(pred, path_to_save):
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
    midi_stream.write('midi', fp=path_to_save + '.mid')


def main_section():
    st.title('Classical Music Generation')
    st.write('')
    st.write('')
    st.image('https://images.pexels.com/photos/697672/pexels-photo-697672.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940',
             use_column_width=True)
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
        path_to_save = st.sidebar.text_input('Select path to save in format drive:\\***\***\\filename')
        if st.sidebar.button('Generate music'):
            pred = gen_pred_tokens(n_starting_points, n_steps, temperature_token, model=model_combined)
            gen_midi_tokens(pred, path_to_save)
            st.success('Successfully generated and saved')
            del model_combined
            gc.collect()

    if model_type == 'Separate models':
        models = load_separate_model()
        n_starting_points = st.sidebar.slider('Choose number of random starting points', 1, 20)
        n_steps = st.sidebar.slider('Choose number of generated steps', 1, 200)
        temperature_note = st.sidebar.slider('Choose temperature note', 0.1, 2.0, 1.0)
        temperature_duration = st.sidebar.slider('Choose temperature duration', 0.1, 2.0, 1.0)
        temperature_offset = st.sidebar.slider('Choose temperature offset', 0.1, 2.0, 1.0)
        path_to_save = st.sidebar.text_input('Select path to save in format drive:\\***\***\\filename')
        if st.sidebar.button('Generate music'):
            pred = gen_pred_separate(n_starting_points, n_steps, temperature_note, temperature_duration, temperature_offset, models=models)
            gen_midi_separate(pred, path_to_save)
            st.success('Successfully generated and saved')
            del models
            gc.collect()

    if model_type == 'Single model separate outputs':
        model = load_model_full()
        n_starting_points = st.sidebar.slider('Choose number of random starting points', 1, 20)
        n_steps = st.sidebar.slider('Choose number of generated steps', 1, 200)
        temperature_note = st.sidebar.slider('Choose temperature note', 0.1, 2.0, 1.0)
        temperature_duration = st.sidebar.slider('Choose temperature duration', 0.1, 2.0, 1.0)
        temperature_offset = st.sidebar.slider('Choose temperature offset', 0.1, 2.0, 1.0)
        path_to_save = st.sidebar.text_input('Select path to save in format drive:\\***\***\\filename')
        if st.sidebar.button('Generate music'):
            pred = gen_pred_full(n_starting_points, n_steps, temperature_note, temperature_duration, temperature_offset, model=model)
            gen_midi_separate(pred, path_to_save)
            st.success('Successfully generated and saved')
            del model
            gc.collect()

main_section()
gc.collect()
