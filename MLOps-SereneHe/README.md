# pic_classification_mnist_v01_xh

Process MNIST files collected in MLOps course: organize them into a single folder, convert them into a single tensor, normalize the tensor, and save this intermediate representation back to the folder for further use.

## Quick Start

### Installation

```bash
# Install dependencies
uv sync
```

### Running the Scripts

The project provides several scripts for the complete ML pipeline:

#### 1. Preprocess Data
Process raw corrupted MNIST files and normalize them:
```bash
# Using invoke task
uvx invoke preprocess-data

# Or run directly
uv run python -m pic_classification_mnist_v01_xh.data data/corruptmnist/raw data/processed
```
This will:
- Load 6 training files and 1 test file from `data/corruptmnist/raw/`
- Concatenate and normalize images (mean=0, std=1)
- Save processed tensors to `data/processed/`

#### 2. Explore Data (Recommended)
Explore and visualize the dataset distribution:
```bash
# Using invoke task
uvx invoke explore

# Or run directly
uv run python -m pic_classification_mnist_v01_xh.explore_data
```
This generates 4 visualizations in `reports/figures/`:
- `class_distribution.png` - Class balance in train/test sets
- `sample_images.png` - Grid of sample images from each class
- `pixel_distribution.png` - Pixel intensity histograms
- `class_statistics.png` - Mean and std per class

#### 3. Train Model
Train a CNN model on the processed data:
```bash
# Using invoke task
uvx invoke train

# Or run directly with custom parameters
uv run python -m pic_classification_mnist_v01_xh.main --lr 0.001 --batch-size 64 --epochs 10
```
This will:
- Train a convolutional neural network (225k parameters)
- Save the trained model to `models/model.pth`
- Generate training curves and save to `reports/figures/training_statistics.png`

#### 4. Evaluate Model
Evaluate the trained model on test data:
```bash
uv run python -m pic_classification_mnist_v01_xh.evaluate models/model.pth
```

#### 5. Visualize Embeddings
Generate t-SNE visualization of model embeddings:
```bash
uv run python -m pic_classification_mnist_v01_xh.visualize models/model.pth
```
This saves the visualization to `reports/figures/embeddings.png`

#### 6. Test Model Architecture
Test the model structure:
```bash
uv run python -m pic_classification_mnist_v01_xh.model
```

### Available Invoke Tasks

```bash
# List all available tasks
uvx invoke --list

# Common tasks
uvx invoke preprocess-data  # Preprocess raw data
uvx invoke explore          # Explore dataset (NEW!)
uvx invoke train            # Train the model
uvx invoke test             # Run tests
uvx invoke docker-build     # Build Docker images
uvx invoke serve-docs       # Serve documentation
```

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
