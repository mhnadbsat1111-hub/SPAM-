# SMS Spam Detection

Comparison of four models for classifying SMS messages as spam or legitimate (ham).
Graduation project, Computer Science, Taif University.

## Dataset
SMS Spam Collection (columns: `Category`, `Message`). Duplicate messages are removed
before splitting, so the same text never appears in both train and test.

- Raw messages: 5,572 (403 duplicates removed, 5,169 remain)
- Spam share: 12.63% (train 3,618 / test 1,551)

Download the CSV and place it at `data/spam.csv`.

## Approach
- One stratified 70/30 split (seed 42) shared by every model
- Models: TF-IDF + Linear SVM, Dense NN (Embedding), GRU, LSTM
- Neural models: early stopping on a validation split; tokenizer fitted on training data only
- Metrics: Accuracy, Precision, Recall, F1 (spam = positive class)

## Results (held-out test set)

| Model               | Accuracy | Precision | Recall | F1 |
|---------------------|----------|-----------|--------|----|
| TF-IDF + Linear SVM | 0.9839   | 0.9831    | 0.8878 | 0.9330 |
| Dense NN            | 0.9781   | 0.9880    | 0.8367 | 0.9061 |
| GRU                 | 0.9826   | 0.9721    | 0.8878 | 0.9280 |
| LSTM                | 0.9826   | 0.9471    | 0.9133 | 0.9299 |

Best model by F1: TF-IDF + Linear SVM (174 of 196 spam caught, 3 ham wrongly flagged). Its confusion matrix is saved to `images/confusion_matrix.png`.

Neural model scores can shift slightly between runs and machines (TensorFlow is not fully deterministic), so re-run `src/train.py` and update the table if your numbers differ.

Note: the best model is picked on the test set, and differences between close models
may be within noise at this dataset size.

## Project Structure
```
├── data/            # place spam.csv here
├── images/          # confusion matrix
├── models/          # saved SVM pipeline (generated)
├── src/
│   ├── train.py     # trains and compares all models
│   └── predict.py   # classify a new message
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py "Congratulations! You won a free prize, call now"
```


## Tech Stack
Python, scikit-learn, TensorFlow/Keras, pandas, matplotlib, seaborn

## Author
Muhannad Al-Thaqafi
