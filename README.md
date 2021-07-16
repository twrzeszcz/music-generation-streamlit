# Genera Info
This is a classical music generation project where we build 2 different architectures in 3 different ways to predict next note, duration and offset. Dataset consisted of 9 violin pieces downloaded from musescore which gives more than 1.5 hours of audio data. Models are explained in the notebooks **prep_train_model_0**, **prep_train_model_1** and **prep_train_model_2**. Music can be generated in the Streamlit app which unfortunately is not deployed cause I could not save the generated *midi* files. However, all the necessary files for deployment are in this repo including the dependencies in the **requirements**, models, pickled unique values and mapping functions and the main file **app_streamlit**.
