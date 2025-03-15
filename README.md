Super-Resolution with ESRGAN (TensorFlow)

This project implements a super-resolution model using a modified ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) architecture in TensorFlow. The model is designed to upsample low-resolution images to high-resolution outputs while preserving realistic details through perceptual loss computed with a pretrained VGG19 network.

Overview

    Generator:
    Uses an ESRGAN-style architecture with Residual-in-Residual Dense Blocks (RRDB) for effective feature extraction and image upscaling. The generator upsamples a 64×64 low-resolution image to a 256×256 high-resolution image using subpixel convolution layers.

    Discriminator:
    A CNN that distinguishes between real high-resolution images and generated images, providing adversarial feedback to improve the generator’s performance.

    Perceptual Loss:
    Leverages features from an intermediate layer of the VGG19 network to compute a content loss that helps the generator produce visually pleasing and realistic images.

    Evaluation Metrics:
    Uses PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) to evaluate the quality of the super-resolved images.


    Dataset

The project uses the DIV2K training dataset


Model Architecture
Generator

    Initial Feature Extraction: Convolutional layer to extract initial features.
    RRDB Blocks: Stacked blocks that use dense connections and residual learning.
    Global Skip Connection: Combines initial features with processed features.
    Upsampling: Two subpixel convolution steps to upscale the image from 64×64 to 256×256.
    Output: A high-resolution image with pixel values in the range [0, 1].

Discriminator

    A CNN composed of multiple convolutional blocks, batch normalization, and LeakyReLU activations.
    The discriminator outputs a probability indicating whether the image is real or generated.

Perceptual Loss

    Uses a pretrained VGG19 network (without top layers) to extract intermediate features.
    The mean squared error between the feature maps of the high-resolution and generated images is computed as the content loss.
