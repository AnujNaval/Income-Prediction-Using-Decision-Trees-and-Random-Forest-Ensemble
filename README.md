# Project README

## Overview

This project implements the full Decision Tree and Random Forest assignment from **COL774 (Assignment 3), Part A**, as described in the course PDF. The goal of this project is to build a decision-tree based classifier from **first principles**, apply one-hot encoding, perform **post-pruning**, compare results against **scikit-learn**, and finally train a **Random Forest** model using grid search and out-of-bag evaluation.

The code supports running any sub-part of the assignment (a–e) using a command-line interface, trains models on the provided UCI Adult Income dataset, generates predictions, saves evaluation plots, and outputs final prediction CSV files for auto-evaluation.

## Project Structure

The project contains a single Python file implementing the entire workflow:

* **Custom Decision Tree Implementation**

  * Handles categorical and continuous attributes
  * Uses entropy + mutual information for splitting
  * Supports multi-way splits for categorical features
  * Median-based threshold splits for continuous features
  * Custom pruning based on validation accuracy

* **One-Hot Encoding Utility**

  * Encodes multi-category categorical attributes

* **Scikit-learn Comparison Models**

  * DecisionTreeClassifier (entropy criterion)
  * RandomForestClassifier (entropy criterion)

* **Plot Generation**

  * Accuracy vs depth plots
  * Pruning progress plots

* **Command Line Driver** (`main()`)

  * Executes one assignment part at a time
  * Saves predictions + plots into the output folder

## How to Run

### **Command Format**

```
python decision_tree.py <train_data_path> <validation_data_path> <test_data_path> <output_folder_path> <question_part>
```

### **Arguments**

| Argument               | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `train_data_path`      | Path to `train.csv`                               |
| `validation_data_path` | Path to `valid.csv`                               |
| `test_data_path`       | Path to `test.csv`                                |
| `output_folder_path`   | Directory where predictions + plots will be saved |
| `question_part`        | One of `a`, `b`, `c`, `d`, `e`                    |

### **Example**

```
python decision_tree.py data/train.csv data/valid.csv data/test.csv outputs a
```

This will:

* Train the custom decision tree for part A
* Evaluate accuracy on train/test
* Save `prediction_a.csv`
* Save `part_a_accuracy.png` plot in the output folder

## Algorithms Used

### **Decision Tree (From Scratch)**

* Computes entropy and mutual information for every candidate split
* Handles:

  * **Categorical attributes** using multi-way splits
  * **Continuous attributes** using median-based splits
* Recursively builds the tree until maximum depth or pure leaf
* Implements prediction by tree traversal

### **One-Hot Encoded Decision Tree (Part B)**

* Replaces categorical features with many categories using one-hot binary features
* Trains deeper trees (depths: 25, 35, 45, 55)

### **Post-Pruning (Part C)**

* Starts with a fully grown tree
* Iteratively prunes nodes
* At each step, removes the node whose pruning produces **maximum validation accuracy gain**
* Stops when no further improvement occurs
* Plots accuracy vs number of nodes

### **Scikit-learn Decision Tree (Part D)**

* Variation over max depth
* Variation over CCP-alpha (post-pruning)
* Compares results with custom implementations

### **Random Forest (Part E)**

* Performs full grid search over:

  * `n_estimators` (50–350)
  * `max_features` (0.1–0.9)
  * `min_samples_split` (2–10)
* Uses **OOB accuracy** to select best model
* Reports train/validation/test accuracies

## Results

This section will include:

* Accuracy plots (train/test vs depth)
* Pruning curves (nodes vs accuracy)
* Comparison between custom DT, pruned DT, sklearn DT, and Random Forest
* Final test accuracies for each part (a–e)

Add your plots and commentary here once the experiments are completed.
