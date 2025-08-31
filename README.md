# Genre Classification ML Pipeline

## Overview

This is an end-to-end machine learning pipeline for music genre classification built with MLflow, Weights & Biases (W&B), and Hydra for configuration management. The pipeline processes music data with audio features and text metadata to predict music genres using a Random Forest classifier.

## Project Architecture

The project follows a modular MLflow pipeline architecture with the following components:

```
genre_classification_ml_pipeline/
├── main.py                 # Main pipeline orchestrator
├── config.yaml            # Central configuration file
├── conda.yml              # Root environment dependencies
├── MLproject              # Root MLflow project file
├── download/              # Data download component
├── preprocess/            # Data preprocessing component
├── check_data/            # Data validation component
├── segregate/             # Data splitting component
├── random_forest/         # Model training component
└── evaluate/              # Model evaluation component
```

## Pipeline Components

### 1. Download Component (`download/`)
- **Purpose**: Downloads raw data from a remote URL and uploads it as a W&B artifact
- **Input**: URL to parquet file
- **Output**: `raw_data.parquet` artifact
- **Key Features**:
  - Streaming download for large files
  - Automatic artifact creation with metadata
  - Temporary file handling for memory efficiency

### 2. Preprocess Component (`preprocess/`)
- **Purpose**: Cleans and transforms raw data
- **Input**: Raw data artifact
- **Output**: `preprocessed_data.csv` artifact
- **Transformations**:
  - Removes duplicate records
  - Creates combined text feature from `title` and `song_name`
  - Handles missing values in text fields

### 3. Check Data Component (`check_data/`)
- **Purpose**: Validates data quality and consistency using pytest
- **Input**: Reference dataset and new sample
- **Tests Performed**:
  - **Schema Validation**: Ensures required columns and data types
  - **Value Range Checks**: Validates audio features are within expected ranges
  - **Class Label Validation**: Confirms only known genre classes are present
  - **Distribution Testing**: Kolmogorov-Smirnov test for data drift detection
- **Supported Genres**:
  ```
  Dark Trap, Underground Rap, Trap Metal, Emo, Rap, RnB, Pop, 
  Hiphop, techhouse, techno, trance, psytrance, trap, dnb, hardstyle
  ```

### 4. Segregate Component (`segregate/`)
- **Purpose**: Splits data into train and test sets
- **Features**:
  - Configurable test size
  - Stratified sampling by genre
  - Reproducible splits with random seed

### 5. Random Forest Component (`random_forest/`)
- **Purpose**: Trains a Random Forest classifier with preprocessing pipeline
- **Pipeline Architecture**:
  ```
  ColumnTransformer:
  ├── Numerical Features: SimpleImputer → StandardScaler
  ├── Categorical Features: SimpleImputer → OrdinalEncoder  
  └── NLP Features: SimpleImputer → TfidfVectorizer
  │
  └── RandomForestClassifier
  ```
- **Features Processed**:
  - **Numerical**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms
  - **Categorical**: time_signature, key
  - **Text**: text_feature (combined title + song_name)

### 6. Evaluate Component (`evaluate/`)
- **Purpose**: Evaluates trained model on test set
- **Metrics**:
  - AUC-ROC (macro-averaged, one-vs-one)
  - Confusion matrix with normalization
- **Outputs**: Performance metrics and visualizations to W&B

## Configuration Management

The pipeline uses Hydra for configuration management with `config.yaml`:

```yaml
main:
  project_name: exercise_14
  experiment_name: dev
  execute_steps: [download, preprocess, check_data, segregate, random_forest, evaluate]
  random_seed: 42

data:
  file_url: "https://github.com/udacity/nd0821-c2-build-model-workflow-exercises/blob/master/lesson-2-data-exploration-and-preparation/exercises/exercise_4/starter/genres_mod.parquet?raw=true"
  reference_dataset: "exercise_14/preprocessed_data.csv:latest"
  ks_alpha: 0.05
  test_size: 0.3
  val_size: 0.3
  stratify: genre

random_forest_pipeline:
  random_forest:
    n_estimators: 100
    max_depth: 13
    class_weight: "balanced"
    # ... other parameters
  tfidf:
    max_features: 10
  features:
    numerical: [danceability, energy, loudness, ...]
    categorical: [time_signature, key]
    nlp: [text_feature]
```

## Setup Instructions

### Prerequisites
- Python 3.10
- Conda/Miniconda
- Weights & Biases account
- MLflow

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd genre_classification_ml_pipeline
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f conda.yml
   conda activate download_data
   ```

3. **Configure W&B**:
   ```bash
   wandb login
   ```

4. **Set environment variables** (optional):
   ```bash
   export WANDB_PROJECT=your_project_name
   export WANDB_RUN_GROUP=your_experiment_name
   ```

## Usage

### Running the Full Pipeline

Execute the complete pipeline with default configuration:
```bash
python main.py
```

### Running Specific Steps

Execute only specific components:
```bash
python main.py main.execute_steps=download,preprocess
```

### Overriding Configuration

Override configuration parameters:
```bash
python main.py data.test_size=0.2 random_forest_pipeline.random_forest.n_estimators=200
```

### Running Individual Components

Each component can be run independently using MLflow:

```bash
# Download data
mlflow run download -P file_url=<url> -P artifact_name=raw_data.parquet

# Preprocess data  
mlflow run preprocess -P input_artifact=raw_data.parquet:latest

# Train model
mlflow run random_forest -P train_data=data_train.csv:latest -P model_config=config.yml
```

## Data Pipeline

### Input Data Schema
The pipeline expects music data with the following features:

**Audio Features** (numerical):
- `danceability` (0-1): How suitable a track is for dancing
- `energy` (0-1): Perceptual measure of intensity and power  
- `loudness` (-35 to 5): Overall loudness in decibels
- `speechiness` (0-1): Presence of spoken words
- `acousticness` (0-1): Confidence measure of whether track is acoustic
- `instrumentalness` (0-1): Predicts whether track contains no vocals
- `liveness` (0-1): Detects presence of an audience
- `valence` (0-1): Musical positiveness conveyed
- `tempo` (50-250): Overall estimated tempo in BPM
- `duration_ms` (20000-1000000): Track duration in milliseconds

**Musical Features** (categorical):
- `time_signature` (1-5): Time signature of the track
- `key` (0-11): Key the track is in

**Text Features**:
- `title`: Song title
- `song_name`: Song name  
- `text_feature`: Engineered feature combining title and song_name

**Target**:
- `genre`: Music genre classification label

### Data Quality Checks

The pipeline includes comprehensive data validation:

1. **Schema Validation**: Ensures all required columns exist with correct data types
2. **Range Validation**: Checks that numerical features fall within expected ranges
3. **Class Validation**: Verifies only known genre labels are present  
4. **Distribution Testing**: Uses Kolmogorov-Smirnov test to detect data drift

## Model Architecture

### Preprocessing Pipeline
- **Numerical Features**: Median imputation → Standard scaling
- **Categorical Features**: Constant imputation → Ordinal encoding
- **Text Features**: Constant imputation → TF-IDF vectorization (binary, max 10 features)

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Class Balancing**: Balanced class weights
- **Feature Selection**: All engineered features used
- **Hyperparameters**: Configurable via YAML (see `config.yaml`)

## Monitoring and Logging

### Weights & Biases Integration
- **Experiment Tracking**: All runs logged with hyperparameters and metrics
- **Artifact Management**: Data and model versioning
- **Visualization**: Feature importance, confusion matrices, performance metrics

### Key Metrics Tracked
- **AUC-ROC**: Macro-averaged one-vs-one for multi-class
- **Feature Importance**: Individual and aggregated NLP importance
- **Confusion Matrix**: Normalized true class predictions
- **Data Drift**: Kolmogorov-Smirnov test p-values

## Model Evaluation

The evaluation component provides comprehensive model assessment:

```python
# Key evaluation metrics
- AUC-ROC Score (macro-averaged, multi-class one-vs-one)
- Normalized confusion matrix  
- Feature importance visualization
- Class-wise performance analysis
```

## Troubleshooting

### Common Issues

1. **Environment Dependencies**:
   ```bash
   # If conda environment conflicts occur
   conda env remove -n download_data
   conda env create -f conda.yml
   ```

2. **W&B Authentication**:
   ```bash
   # Re-authenticate with W&B
   wandb login --relogin
   ```

3. **Data Download Issues**:
   - Check internet connection
   - Verify URL accessibility
   - Check disk space for large files

4. **Memory Issues**:
   - Reduce batch size in configuration
   - Use streaming for large datasets
   - Monitor system resources

### Error Handling

The pipeline includes robust error handling:
- Automatic cleanup of temporary files
- Graceful handling of missing data
- Comprehensive logging at each step
- Artifact validation before processing

## Development Guidelines

### Adding New Components

1. Create new directory with `MLproject`, `conda.yml`, and implementation
2. Update main pipeline configuration
3. Add appropriate tests
4. Document component interface

### Modifying Existing Components

1. Update relevant `conda.yml` for new dependencies
2. Maintain backward compatibility with artifacts
3. Update configuration schema if needed
4. Test end-to-end pipeline

### Best Practices

- Use semantic versioning for artifacts
- Include comprehensive logging
- Implement proper error handling
- Follow consistent code style
- Document configuration changes


## License and Contributing
This project is part of the Udacity Machine Learning DevOps Engineer Nanodegree program. The code and materials are provided for educational purposes.
