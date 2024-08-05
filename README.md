# 🌟 Fashion Recommendation System Using Deep Learning 🌟

Welcome to the **Fashion Recommendation System Using Deep Learning** project! This repository contains the code and resources for building an AI-powered fashion recommendation system.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Explanation](#explanation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
👗 **Fashion Recommendation System Using Deep Learning** leverages the power of deep learning to suggest similar fashion items based on an input image. Utilizing the ResNet50 model, this project preprocesses images, extracts features, and finds the nearest neighbors to recommend visually similar fashion items.

## Features
- 🌈 **Image Preprocessing**: Resize and preprocess images for model input.
- 🧠 **Deep Learning Model**: Use ResNet50 for feature extraction.
- 🔍 **Similarity Search**: Implement nearest neighbors search to find similar items.
- 🖼️ **Visualization**: Display recommended fashion items.

## Usage

1. **Preprocess Images**: Load and preprocess the images to match the input shape required by ResNet50.
2. **Extract Features**: Use the modified ResNet50 model to extract and normalize feature vectors from the images.
3. **Store Features**: Save the extracted features and their corresponding filenames.
4. **Find Similar Items**: Use the NearestNeighbors algorithm to find and display similar items based on an input image.


## Explanation

### Model Setup

- **Pre-trained ResNet50**: A ResNet50 model pre-trained on ImageNet is loaded and modified by removing the top layer and adding a global max pooling layer to extract feature vectors.
- **Model Summary**: The modified model's architecture is summarized.

### Image Processing

- **Loading and Preprocessing**: Images are loaded using OpenCV, resized to 224x224 pixels, and converted from BGR to RGB format.
- **Normalization**: Feature vectors extracted from images are normalized for consistency.

### Feature Extraction

- **Function for Feature Extraction**: A function is defined to preprocess an image, pass it through the model, and normalize the resulting feature vector.

### Feature Storage

- **Feature List Creation**: Features are extracted from all images in the dataset and stored in a list along with their filenames.
- **Saving Features**: The feature list and filenames are saved using pickle for later use.

### Nearest Neighbors Search

- **Finding Similar Images**: The NearestNeighbors algorithm from scikit-learn is used to find the closest matches to an input image based on the extracted features.
- **Displaying Results**: The input image and its nearest neighbors are displayed side by side using Matplotlib.

## Contributing

Contributions are welcome! Please open an issue or create a pull request if you have any suggestions for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For more information, feel free to reach out to [ankitroy6321@gmail.com].

Happy coding!


