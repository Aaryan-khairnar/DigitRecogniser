# Digit Recognizer in C++

## Overview
This is a **digit recognizer** implemented in C++. The program processes the **MNIST dataset**, applies a **classification model**, and predicts handwritten digits.

## Features
- Reads CSV input files containing **28x28 pixel images** represented as numerical values.
- Implements **a machine learning model** for digit classification.
- Outputs predictions in a CSV format for submission.
- Achieved a **Kaggle score of 0.72264**.

## Requirements
- **C++ Compiler** (GCC, Clang, MSVC, etc.)
- **Standard Template Library (STL)** for handling input/output
- CSV file handling utilities

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/digit-recognizer.git
   cd digit-recognizer
   ```
2. Compile the program:
   ```bash
   g++ -o recognizer main.cpp
   ```

## Usage
- **Input:** A CSV file with pixel values representing handwritten digits.
- **Processing:** The model reads and processes the images.
- **Output:** A CSV file with predicted digit labels.

## Results
- **Kaggle Score:** 0.72264
- **Performance:** The model achieved decent accuracy on the test dataset.

## Future Improvements
- Implementing a **Neural Network** for better accuracy.
- Optimizing feature extraction and preprocessing.
- Adding support for real-time digit recognition.

## License
MIT License. Feel free to use and improve it!