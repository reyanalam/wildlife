# Wildlife Image Generation using GANs


This repository contains a Generative Adversarial Network (GAN) architecture designed to create realistic synthetic wildlife images. The model is trained on a dataset consisting of 25 different classes of wildlife, and the generator is capable of producing new, unique images inspired by these classes.

## Overview

GANs, or Generative Adversarial Networks, are a type of neural network architecture consisting of two competing networks:

1. **Generator**: Creates synthetic data that resembles the training data.
2. **Discriminator**: Distinguishes between real and generated data.

Through iterative training, the generator improves its ability to create realistic data, while the discriminator gets better at identifying fake samples. In this project, the GAN is trained to generate wildlife images based on the provided dataset.

## Features

- **25 Wildlife Classes**: The model is trained on a diverse dataset representing 25 different wildlife classes, ensuring a variety of generated images.
- **Custom Generator**: Designed to synthesize high-quality images by learning from the patterns in the dataset.
- **Scalable Architecture**: Easily adaptable to include more classes or different types of data.

## Dataset

The dataset includes images from 25 wildlife categories such as lions, tigers, elephants, birds, and others. These images are preprocessed and used for training the GAN model.

## GAN Architecture

### Generator
The generator takes random noise as input and generates synthetic images. It consists of multiple layers of transposed convolutions,  and LeakyReLU activations to ensure high-quality image synthesis.

### Discriminator
The discriminator classifies images as real (from the dataset) or fake (produced by the generator). It uses convolutional layers, and LeakyReLU activations for robust classification.

### Loss Function
Both the generator and discriminator are trained using the adversarial loss function:
- **Generator Loss**: Measures how well the generator fools the discriminator.
- **Discriminator Loss**: Measures how well the discriminator distinguishes real from fake images.

## Results

The GAN successfully generated synthetic wildlife images that closely resemble the training data. The generator is able to create unique images that blend features from multiple classes, demonstrating the model's ability to generalize.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wildlife-gan.git
   cd wildlife-gan
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a GPU-enabled environment for faster training (optional but recommended).

## Training the Model

Follow these steps to train the GAN:

1. Prepare the dataset:
   - Organize the dataset into folders, each representing a class.

2. Run the training script:
   ```bash
   python main.py 
   ```

   Replace `<num_epochs>` with the desired number of epochs, `<batch_size>` with the batch size, and `<path_to_dataset>` with the path to your dataset.

3. Monitor training progress:
   - Check the loss values for the generator and discriminator.
   - View generated images at regular intervals to evaluate quality.

4. Save the trained models:
   - The generator's generated image will be saved in the `generated_images/` directory after training.

5. Run the `generator.py` script:
   ```bash
   python generator.py 
   ```

   Replace `<output_directory>` with the path where the generated images should be saved and `<number_of_images>` with the desired number of images to generate.

6. The generated images will be saved in the specified output directory.


