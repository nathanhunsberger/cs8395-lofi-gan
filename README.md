# 8-Bit Lofi Music Generation with an RNN-GAN
Authors: Berke Lunstad, Nathan Hunsberger, Shivam Vohra

Vanderbilt University, CS 8395

This repository is the codebase for our Final Project for CS 8395: Representation Learning. This project uses an encoding of sound that use Fast Fourier Transforms to encode wav files as csvs and then train an RNN-GAN on this csv data. We used https://github.com/seyedsaleh/music-generator/tree/master as inspiration. Our resulting project is a GAN that can generate 8-Bit Lofi style wav files, as can be seen in the `results/` directory. Read our paper in the `CS_8395_Final_Report.pdf` file to learn more and see our references. To see some failed attempts at encoding sound, checking  the `failed_encodings/`directory.

## Using this Project
To use this project, the encoder can be ran with `python3 encoder.py`. You must modify `encoder.py` to include the specific wav file you wish to encode. Once you have run the encoder, you can the resulting csvs and `Music_Generating_using_C_RNN_GAN_Model_Report.ipynb` to train an RNN-GAN on your encoded music. Once finished training, the ipynb has cells that create csvs from the Generator output. Save those csv files and modify `decoder.py` to decode those csvs, and run `python3 decoder.py` to generate resulting wav files.