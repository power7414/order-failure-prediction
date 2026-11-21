# Will an online order fail?

Team project (6 members) for *Big Data and Business Analytics*, National Taiwan University, spring 2025.

## Question
Using 3.4 million e-commerce orders from 91APP (about 11.9% fail), can shopping behavior **before** an order signal that it will fail?

## Approach
- Features: counts of 18 behavior types in the 72 hours before each order, plus order, payment, shipping and member attributes.
- Models: price-stratified XGBoost (teammates) and deep models — DNN, LSTM, GRU (my part, Keras).
- Class imbalance handled with class weights; evaluated by AUC and recall.
- This notebook contains the data preparation, DNN and LSTM; the GRU run and the team's XGBoost are reported in the final report but not included here.

## Results (AUC; 0.5 is chance)
| Model | AUC |
|---|---|
| DNN | **0.850** |
| DNN, orders with behavior records | 0.858 |
| LSTM | 0.705 (highest recall, 0.77) |
| GRU | 0.704 |

## Data
The 91APP dataset was provided for the course and is **not included**. The notebook expects CSV files under `data/`; outputs were cleared.
