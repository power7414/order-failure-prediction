# Will an online order fail?

Team project (6 members) for *Big Data and Business Analytics*, National Taiwan University, spring 2025.

## Question
Using 3.4 million e-commerce orders from 91APP (about 11.9% fail), can shopping behavior **before** an order signal that it will fail?

## Approach
- Features: counts of 18 behavior types in the 72 hours before each order, plus order, payment, shipping and member attributes.
- Team models: price-stratified XGBoost (teammates) and deep models — DNN, LSTM, GRU (my part, Keras).

## Results reported in the team's final report (AUC; 0.5 is chance)
| Model | AUC |
|---|---|
| DNN | **0.850** |
| DNN, orders with behavior records | 0.858 |
| LSTM | 0.705 (highest recall, 0.77) |
| GRU | 0.704 |

## What is in this notebook
`order_failure_models.ipynb` is my exploratory model, not the exact runs behind the table: a two-branch network with a **bidirectional LSTM over the 72-hour behavior sequence** and a dense branch over static order features, trained on a **random sample of 50,000 orders** (to keep sequence building tractable) with class weights for the imbalance. It also includes a perturbation check of how much the behavior sequence contributes to AUC. The DNN, GRU and XGBoost runs from the report are not included.

Limitations of this notebook: the event times inside each 72-hour window are simulated with `np.random.uniform`, so the LSTM cannot learn real ordering; behavior codes are fed as numbers rather than embeddings; and clipping and scaling are fitted before the split. Treat it as an exploratory model, not as the source of the reported AUC.

## Data
The 91APP dataset was provided for the course and is **not included**. The notebook expects CSV files under `data/`; outputs were cleared.
