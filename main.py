import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, LeakyReLU, Dropout, Reshape
from keras.optimizers import Adam
import os

image_directory = r"C:\Users\Reyan\Desktop\Projects\GAN_wildlife\data\animals\animals"  

datagen = ImageDataGenerator(rescale=1./255)  # Normalize images to [0, 1]

train_data = datagen.flow_from_directory(
    image_directory,
    target_size=(256, 256),  # Resize all images to a fixed size
    batch_size=32,
    class_mode=None,  # For GANs, you don't need class labels, just images
    shuffle=True
)

images=[]
for image in train_data:
    images.append(image)
    if len(images)*train_data.batch_size >= train_data.samples:
        break
x_train = np.concatenate(images, axis=0)

def discriminator(in_shape=(256, 256, 3)):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model


def generator(dim):
    
    model = Sequential()
    nodes = 32 * 32 * 256
    model.add(Dense(nodes, input_dim=dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((32, 32, 256)))
    
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    return model

cross_entropy = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tensorflow.ones_like(real_output), real_output)  # Real images labeled as 1
    fake_loss = cross_entropy(tensorflow.zeros_like(fake_output), fake_output)  # Fake images labeled as 0
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tensorflow.ones_like(fake_output), fake_output)  # Generator wants fake images to be labeled as 1

# Optimizers
discriminator_optimizer = Adam(1e-4)
generator_optimizer = Adam(1e-4)

def train_step(real_images, generator, discriminator, batch_size, latent_dim):
    noise = tensorflow.random.normal([batch_size, latent_dim])  # Latent vector
    fake_images = generator(noise, training=True)

    with tensorflow.GradientTape() as tape_d:
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        d_loss = discriminator_loss(real_output, fake_output)

    discriminator_gradients = tape_d.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with tensorflow.GradientTape() as tape_g:
        fake_images = generator(noise, training=True)  
        fake_output = discriminator(fake_images, training=True)
        g_loss = generator_loss(fake_output)

    generator_gradients = tape_g.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    return d_loss, g_loss

def save_generated_samples(generator, latent_dim, epoch, output_dir="generated_images", n_samples=16):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        latent_points = np.random.randn(n_samples, latent_dim)
        if latent_points is None:
            raise ValueError("Latent points generation failed")
        
        generated_samples = generator.predict(latent_points)

        for i in range(n_samples):
            img = generated_samples[i, :, :, :]
            img = np.array(img * 255, dtype=np.uint8)  # Convert to [0, 255] range for saving
            
            img_path = os.path.join(output_dir, f"generated_epoch_{epoch+1}_sample_{i+1}.png")
            plt.imsave(img_path, img)
        
        print(f"Images saved in directory: {output_dir}")
    except Exception as e:
        print(f"Error in save_generated_samples: {e}")

def train(generator, discriminator, dataset, batch_size, latent_dim, epochs):
    for epoch in range(epochs):
        for i in range(0, dataset.shape[0], batch_size):
            real_images = dataset[i:i + batch_size]
            d_loss, g_loss = train_step(real_images, generator, discriminator, batch_size, latent_dim)

        print(f"Epoch: {epoch+1}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_generated_samples(generator, latent_dim, epoch + 1)

latent_dim = 10
dis_model = discriminator()
gen_model = generator(latent_dim)

epochs = 70
batch_size=32
if x_train is not None:
    train(gen_model, dis_model, x_train, batch_size, latent_dim, epochs)
else:
    print("Dataset loading failed, training will not proceed.")

gen_model.save("generator_model.h5")
print("Generator model saved successfully!")