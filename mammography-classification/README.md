# README.md

# Mammography Classification Project

This project aims to develop classification models for mammography images, specifically focusing on BI-RADS categories and lesion detection.

## Overview

The Mammography Classification Project includes data preprocessing, model training, and evaluation components. It utilizes machine learning techniques to classify mammography images, providing insights into BI-RADS categories and lesion characteristics.

## Project Structure

```
mammography-classification
├── src
│   ├── data
│   ├── models
│   ├── utils
│   └── train.py
├── notebooks
├── tests
├── config
└── requirements.txt
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mammography-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- To preprocess the data, run:
  ```
  python src/data/preprocessing.py
  ```

- To train the models, execute:
  ```
  python src/train.py
  ```

- For exploratory data analysis, open the Jupyter notebook:
  ```
  notebooks/exploratory_analysis.ipynb
  ```

- To evaluate the models, use:
  ```
  notebooks/model_evaluation.ipynb
  ```

## Testing

Run the unit tests with:
```
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License.