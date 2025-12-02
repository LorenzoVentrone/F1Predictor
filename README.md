# F1 Predictor 🏎️

A machine learning project to predict Formula 1 race outcomes by classifying driver finishes into performance groups (Podium, Points, No Points). 

## 🎯 Project Aim

This project aims to predict Formula 1 race results using historical data from 1980 to 2025.  The prediction task is a **multi-class classification** problem where drivers are classified into one of three finish groups:

- **Podium**: Finished in positions 1-3
- **Points**: Finished in positions 4-10
- **No Points**: Finished outside the top 10 or did not finish

## 📊 Dataset

The project uses the [Formula 1 Race Data](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) dataset from Kaggle, which contains comprehensive F1 statistics including:

- Race results
- Driver standings
- Constructor standings
- Qualifying results
- Circuit information

## 🔧 Preprocessing

### Data Loading & Merging
Multiple CSV files are loaded and merged to create a comprehensive dataset:
- `races.csv` - Race information
- `results.csv` - Race results
- `drivers.csv` - Driver details
- `constructors.csv` - Team information
- `driver_standings.csv` - Championship standings
- `constructor_standings.csv` - Constructor standings
- `qualifying. csv` - Qualifying positions
- `status.csv` - Finish status

### Data Filtering
- Only races from **1980 onwards** are included
- Regulation eras are categorized to capture different F1 rule periods:
  - Ground Effect Era (1980-1982)
  - Turbo Era (1983-1988)
  - Early NA (1989-1994)
  - Post-Imola (1995-1999)
  - V10 Era (2000-2005)
  - V8 Era (2006-2009)
  - Hybrid V1 (2010-2013)
  - High Aero (2014-2016)
  - Ground Effect (2017-2021)
  - Post-2022 (2022-2025)

### Target Variable Creation
The `FinishGroup` target is created using position binning:
- Positions 1-3 → "Podium"
- Positions 4-10 → "Points"
- Positions 11+ or DNF → "No Points"

### Missing Value Handling
- `driver_points`: filled with 0
- `driver_position`: filled with 30 (unranked)
- `driver_wins`: filled with 0
- `constructor_points`: filled with 0
- `constructor_position`: filled with 20 (unranked)
- `constructor_wins`: filled with 0
- `qualifying_position`: filled with 20 (back of grid)

## 🧠 Feature Set

### Numerical Features
| Feature | Description |
|---------|-------------|
| `year` | Season year |
| `driver_points` | Driver's championship points |
| `driver_position` | Driver's championship position |
| `driver_wins` | Driver's wins in the season |
| `constructor_points` | Team's championship points |
| `constructor_position` | Team's championship position |
| `constructor_wins` | Team's wins in the season |
| `qualifying_position` | Starting grid position |

### Categorical Features
| Feature | Description |
|---------|-------------|
| `constructor_name` | Team name |
| `race_name` | Grand Prix name |
| `regulation_eras` | F1 regulation period |
| `did_finish` | Binary indicator if driver finished the race |
| `driver` | Driver reference name |

### Feature Transformations
- **Numerical features**: StandardScaler normalization
- **Categorical features**: Target Encoding

## 📈 Data Splitting

The dataset is split using **stratified sampling** to maintain class distribution:

| Split | Size | Samples |
|-------|------|---------|
| Training | 64% | 12,320 |
| Validation | 16% | 3,080 |
| Test | 20% | 3,851 |

**Cross-validation**: 5-fold Stratified K-Fold with shuffle (random_state=42)

## 🤖 Machine Learning Models

The following classifiers are implemented and compared:

1. **Gaussian Naive Bayes** (`GaussianNB`)
2. **Logistic Regression** (`LogisticRegression`)
3.  **Decision Tree Classifier** (`DecisionTreeClassifier`)
4. **Random Forest Classifier** (`RandomForestClassifier`)
5. **Support Vector Machine** (`SVC`)
6. **One-vs-One SVM** (`OneVsOneClassifier`)

### Hyperparameter Tuning
- `RandomizedSearchCV` is used for hyperparameter optimization
- Parameter distributions include `loguniform`, `uniform`, and `randint` from scipy

## 📊 Model Evaluation

Models are evaluated using:

- **Accuracy Score**: Overall classification accuracy
- **F1 Score**: Weighted F1 score for imbalanced classes
- **Classification Report**: Precision, recall, F1 per class
- **Confusion Matrix**: Visual representation via `ConfusionMatrixDisplay`
- **ROC Curves**: Multi-class ROC curves with AUC scores using One-vs-Rest binarization
- **Learning Curves**: Training vs validation performance across sample sizes

### Class Distribution
| Class | Proportion |
|-------|------------|
| No Points | 58.0% |
| Points | 29.2% |
| Podium | 12.7% |

## 🛠️ Requirements

```bash
pip install kagglehub pandas numpy matplotlib seaborn category_encoders scikit-learn
```

### Dependencies
- `kagglehub` - Dataset download
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` & `seaborn` - Visualization
- `scikit-learn` - Machine learning models and utilities
- `category_encoders` - Target encoding
- `scipy` - Statistical distributions for hyperparameter tuning

## 🚀 Usage

1. Open the notebook in Google Colab or Jupyter
2.  Run all cells sequentially
3. The dataset will be automatically downloaded from Kaggle
4. Models will be trained and evaluated

## 📁 Project Structure

```
F1Predictor/
├── F1_predictor. ipynb    # Main Jupyter notebook
└── README.md             # Project documentation
```

## 📝 License

This project uses publicly available Formula 1 data from Kaggle. 

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. 
