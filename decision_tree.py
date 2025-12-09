import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

class Node:
    """Node class for the decision tree."""
    def __init__(self, feature=None, threshold=None, value=None, children=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.children = children

class DecisionTree:
    """Custom Decision Tree classifier."""
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.categorical_cols = []
        self.continuous_cols = []

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _mutual_information(self, X_col, y, threshold=None):
        if threshold is not None:  # Continuous feature
            left_mask = X_col <= threshold
            right_mask = X_col > threshold
            subsets = [y[left_mask], y[right_mask]]
        else:  # Categorical feature
            categories = np.unique(X_col)
            subsets = [y[X_col == cat] for cat in categories]

        parent_entropy = self._entropy(y)
        n = len(y)
        child_entropy = 0
        for subset in subsets:
            if len(subset) == 0:
                continue
            child_entropy += (len(subset) / n) * self._entropy(subset)
        return parent_entropy - child_entropy

    def _find_best_split(self, X, y):
        best_info_gain = -1
        best_feature = None
        best_threshold = None

        for col in X.columns:
            if X[col].dropna().empty:
                continue

            if col in self.categorical_cols:
                info_gain = self._mutual_information(X[col], y)
                threshold = None
            else:
                threshold = np.median(X[col])
                if np.isnan(threshold):
                    continue
                info_gain = self._mutual_information(X[col], y, threshold)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = col
                best_threshold = threshold if col in self.continuous_cols else None

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        y = y.astype(int)
        if depth == self.max_depth or len(np.unique(y)) == 1:
            counts = np.bincount(y)
            return Node(value=np.argmax(counts) if len(counts) > 0 else 0)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            counts = np.bincount(y)
            return Node(value=np.argmax(counts) if len(counts) > 0 else 0)

        node = Node(feature=best_feature, threshold=best_threshold)

        if best_feature in self.categorical_cols:
            categories = X[best_feature].unique()
            node.children = {}
            for cat in categories:
                mask = X[best_feature] == cat
                if mask.sum() == 0:
                    node.children[cat] = Node(value=0)
                else:
                    X_sub = X[mask].drop(columns=best_feature)
                    y_sub = y[mask]
                    node.children[cat] = self._build_tree(X_sub, y_sub, depth+1)
        else:
            left_mask = X[best_feature] <= best_threshold
            right_mask = X[best_feature] > best_threshold
            node.children = {
                'left': self._build_tree(X[left_mask], y[left_mask], depth+1) if left_mask.sum() > 0 else Node(value=0),
                'right': self._build_tree(X[right_mask], y[right_mask], depth+1) if right_mask.sum() > 0 else Node(value=0)
            }

        return node

    def fit(self, X, y, categorical_cols):
        self.categorical_cols = categorical_cols
        self.continuous_cols = [col for col in X.columns if col not in categorical_cols]
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if node.feature in self.categorical_cols:
            category = x[node.feature]
            if category not in node.children:
                return 0
            return self._predict_sample(x, node.children[category])
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_sample(x, node.children['left'])
            else:
                return self._predict_sample(x, node.children['right'])

    def predict(self, X):
        return [self._predict_sample(row, self.tree) for _, row in X.iterrows()]

    def _count_nodes(self, node):
        """Count the number of nodes in the tree."""
        if node is None:
            return 0
        if node.value is not None:  # Leaf node
            return 1
        count = 1  # Current node
        if node.children is not None:
            if isinstance(node.children, dict):  # Categorical split
                for child in node.children.values():
                    count += self._count_nodes(child)
            else:  # Continuous split
                count += self._count_nodes(node.children['left'])
                count += self._count_nodes(node.children['right'])
        return count

    def _get_all_nodes(self, node):
        """Get all non-leaf nodes in the tree."""
        nodes = []
        if node is None or node.value is not None:
            return nodes
        nodes.append(node)
        if node.children is not None:
            if isinstance(node.children, dict):  # Categorical split
                for child in node.children.values():
                    nodes.extend(self._get_all_nodes(child))
            else:  # Continuous split
                nodes.extend(self._get_all_nodes(node.children['left']))
                nodes.extend(self._get_all_nodes(node.children['right']))
        return nodes

    def _prune_node(self, node, path):
        """Prune a specific node in the tree."""
        if not path:
            return
        current = path[0]
        if len(path) == 1:  # We've reached the node to prune
            counts = self._get_class_counts(node)
            node.value = np.argmax(counts) if len(counts) > 0 else 0
            node.feature = None
            node.threshold = None
            node.children = None
        else:
            if isinstance(node.children, dict):  # Categorical split
                if current in node.children:
                    self._prune_node(node.children[current], path[1:])
            else:  # Continuous split
                if current in node.children:
                    self._prune_node(node.children[current], path[1:])

    def _get_class_counts(self, node):
        """Get class counts for all samples under a node."""
        if node.value is not None:  # Leaf node
            return np.bincount([node.value])
        counts = np.zeros(2, dtype=int)
        if node.children is not None:
            if isinstance(node.children, dict):  # Categorical split
                for child in node.children.values():
                    counts += self._get_class_counts(child)
            else:  # Continuous split
                counts += self._get_class_counts(node.children['left'])
                counts += self._get_class_counts(node.children['right'])
        return counts

    def prune(self, X_val, y_val):
        """Prune the tree to improve validation accuracy."""
        best_accuracy = accuracy_score(y_val, self.predict(X_val))
        improved = True
        
        while improved:
            improved = False
            nodes = self._get_all_nodes(self.tree)
            best_node_to_prune = None
            best_path_to_prune = None
            best_new_accuracy = best_accuracy
            
            # Try pruning each node and keep track of the best one
            for node in nodes:
                # Make a copy of the tree to test pruning this node
                original_tree = deepcopy(self.tree)
                path = self._find_path_to_node(self.tree, node)
                
                # Prune the node in the copied tree
                temp_tree = deepcopy(self.tree)
                self._prune_node(temp_tree, path)
                
                # Test accuracy
                original_tree_backup = self.tree
                self.tree = temp_tree
                current_accuracy = accuracy_score(y_val, self.predict(X_val))
                self.tree = original_tree_backup
                
                if current_accuracy > best_new_accuracy:
                    best_new_accuracy = current_accuracy
                    best_node_to_prune = node
                    best_path_to_prune = path
            
            # If we found a node that improves accuracy, prune it
            if best_new_accuracy > best_accuracy:
                self._prune_node(self.tree, best_path_to_prune)
                best_accuracy = best_new_accuracy
                improved = True

    def _find_path_to_node(self, current_node, target_node, path=None):
        """Find the path to a specific node in the tree."""
        if path is None:
            path = []
        
        if current_node == target_node:
            return path
        
        if current_node.children is None:
            return None
        
        if isinstance(current_node.children, dict):  # Categorical split
            for category, child in current_node.children.items():
                result = self._find_path_to_node(child, target_node, path + [category])
                if result is not None:
                    return result
        else:  # Continuous split
            for direction in ['left', 'right']:
                result = self._find_path_to_node(current_node.children[direction], target_node, path + [direction])
                if result is not None:
                    return result
        return None

def load_data(train_path, valid_path, test_path):
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)

    valid_income = ['<=50K', '>50K']
    train = train[train['income'].str.strip().isin(valid_income)]
    valid = valid[valid['income'].str.strip().isin(valid_income)]
    test = test[test['income'].str.strip().isin(valid_income)]

    y_train = train['income'].str.strip().map({'<=50K': 0, '>50K': 1}).values.astype(int)
    y_valid = valid['income'].str.strip().map({'<=50K': 0, '>50K': 1}).values.astype(int)
    y_test = test['income'].str.strip().map({'<=50K': 0, '>50K': 1}).values.astype(int)

    X_train = train.drop(columns=['income'])
    X_valid = valid.drop(columns=['income'])
    X_test = test.drop(columns=['income'])

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def one_hot_encode_data(X_train, X_valid, X_test):
    """Apply one-hot encoding to categorical variables with more than 2 categories"""
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Identify categorical columns with more than 2 categories
    multi_cat_cols = [col for col in categorical_cols if len(X_train[col].unique()) > 2]
    
    if not multi_cat_cols:
        return X_train, X_valid, X_test, categorical_cols
    
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform training data
    train_encoded = encoder.fit_transform(X_train[multi_cat_cols])
    valid_encoded = encoder.transform(X_valid[multi_cat_cols])
    test_encoded = encoder.transform(X_test[multi_cat_cols])
    
    # Get feature names
    feature_names = encoder.get_feature_names_out(multi_cat_cols)
    
    # Create DataFrames from encoded data
    train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
    valid_encoded_df = pd.DataFrame(valid_encoded, columns=feature_names, index=X_valid.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)
    
    # Drop original columns and concatenate with encoded data
    X_train = pd.concat([X_train.drop(columns=multi_cat_cols), train_encoded_df], axis=1)
    X_valid = pd.concat([X_valid.drop(columns=multi_cat_cols), valid_encoded_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=multi_cat_cols), test_encoded_df], axis=1)
    
    # Update categorical columns list (only those with <= 2 categories remain)
    categorical_cols = [col for col in categorical_cols if col not in multi_cat_cols]
    
    return X_train, X_valid, X_test, categorical_cols

def part_a(train_path, valid_path, test_path, output_folder):
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(train_path, valid_path, test_path)
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    depths = [5, 10, 15, 20]
    train_accs, test_accs = [], []
    final_model = None

    for depth in depths:
        clf = DecisionTree(max_depth=depth)
        clf.fit(X_train, y_train, categorical_cols)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Depth {depth}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        if depth == 20:
            final_model = clf

    # Save predictions
    pd.DataFrame({'prediction': final_model.predict(X_test)}).to_csv(os.path.join(output_folder, 'prediction_a.csv'), index=False)

    # Plot
    plt.plot(depths, train_accs, label='Train')
    plt.plot(depths, test_accs, label='Test')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'part_a_accuracy.png'))
    plt.close()

def part_b(train_path, valid_path, test_path, output_folder):
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(train_path, valid_path, test_path)
    
    # Apply one-hot encoding
    X_train, X_valid, X_test, categorical_cols = one_hot_encode_data(X_train, X_valid, X_test)
    
    depths = [25, 35, 45, 55]
    train_accs, test_accs = [], []
    final_model = None

    for depth in depths:
        clf = DecisionTree(max_depth=depth)
        clf.fit(X_train, y_train, categorical_cols)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Depth {depth}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        if depth == 55:
            final_model = clf

    # Save predictions
    pd.DataFrame({'prediction': final_model.predict(X_test)}).to_csv(os.path.join(output_folder, 'prediction_b.csv'), index=False)

    # Plot
    plt.plot(depths, train_accs, label='Train')
    plt.plot(depths, test_accs, label='Test')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'part_b_accuracy.png'))
    plt.close()

def part_c(train_path, valid_path, test_path, output_folder):
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(train_path, valid_path, test_path)
    
    # Apply one-hot encoding (same as part b)
    X_train, X_valid, X_test, categorical_cols = one_hot_encode_data(X_train, X_valid, X_test)
    
    depths = [25, 35, 45, 55]
    
    for depth in depths:
        print(f"\nPruning for max_depth={depth}")
        # Train initial tree
        clf = DecisionTree(max_depth=depth)
        clf.fit(X_train, y_train, categorical_cols)
        
        # Track metrics during pruning
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        node_counts = []
        
        # Initial metrics
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_valid, clf.predict(X_valid))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        node_count = clf._count_nodes(clf.tree)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        node_counts.append(node_count)
        
        print(f"Initial - Nodes: {node_count}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Perform pruning until no improvement
        best_val_acc = val_acc
        improved = True
        
        while improved:
            improved = False
            nodes = clf._get_all_nodes(clf.tree)
            best_node_to_prune = None
            best_new_val_acc = best_val_acc
            
            # Try pruning each node and keep track of the best one
            for node in nodes:
                # Make a copy of the tree to test pruning this node
                temp_tree = deepcopy(clf)
                path = temp_tree._find_path_to_node(temp_tree.tree, node)
                temp_tree._prune_node(temp_tree.tree, path)
                
                # Test validation accuracy
                current_val_acc = accuracy_score(y_valid, temp_tree.predict(X_valid))
                
                if current_val_acc > best_new_val_acc:
                    best_new_val_acc = current_val_acc
                    best_node_to_prune = node
            
            # If we found a node that improves validation accuracy, prune it
            if best_new_val_acc > best_val_acc:
                path = clf._find_path_to_node(clf.tree, best_node_to_prune)
                clf._prune_node(clf.tree, path)
                best_val_acc = best_new_val_acc
                improved = True
                
                # Record metrics after pruning
                train_acc = accuracy_score(y_train, clf.predict(X_train))
                val_acc = best_val_acc
                test_acc = accuracy_score(y_test, clf.predict(X_test))
                node_count = clf._count_nodes(clf.tree)
                
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                test_accuracies.append(test_acc)
                node_counts.append(node_count)
                
                print(f"Pruned - Nodes: {node_count}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Plot the pruning process for this depth
        plt.figure()
        plt.plot(node_counts, train_accuracies, label='Train Accuracy')
        plt.plot(node_counts, val_accuracies, label='Validation Accuracy')
        plt.plot(node_counts, test_accuracies, label='Test Accuracy')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Accuracy')
        plt.title(f'Pruning Progress (max_depth={depth})')
        plt.legend()
        plt.gca().invert_xaxis()  # Reverse x-axis to show pruning progression
        plt.savefig(os.path.join(output_folder, f'part_c_pruning_depth_{depth}.png'))
        plt.close()
        
        # Save final predictions for this depth
        pd.DataFrame({'prediction': clf.predict(X_test)}).to_csv(
            os.path.join(output_folder, f'prediction_c_depth_{depth}.csv'), 
            index=False
        )

def part_d(train_path, valid_path, test_path, output_folder):
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(train_path, valid_path, test_path)
    
    # Apply one-hot encoding (same as part b)
    X_train, X_valid, X_test, categorical_cols = one_hot_encode_data(X_train, X_valid, X_test)
    
    # Convert remaining categorical columns to numerical using label encoding
    from sklearn.preprocessing import LabelEncoder
    remaining_categorical = X_train.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_categorical:
        le = LabelEncoder()
        # Fit on combined data to handle all possible categories
        combined = pd.concat([X_train[col], X_valid[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_valid[col] = le.transform(X_valid[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Part d(i): Vary max_depth with scikit-learn
    depths = [25, 35, 45, 55]
    train_accs, val_accs, test_accs = [], [], []
    best_depth = None
    best_accuracy = 0
    
    print("\nPart d(i): Varying max_depth with scikit-learn")
    for depth in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_valid, clf.predict(X_valid))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        
        print(f"Depth {depth}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_depth = depth
    
    # Plot accuracy vs max_depth
    plt.figure()
    plt.plot(depths, train_accs, label='Train Accuracy')
    plt.plot(depths, val_accs, label='Validation Accuracy')
    plt.plot(depths, test_accs, label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Scikit-learn Decision Tree: Accuracy vs Max Depth')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'part_d_max_depth.png'))
    plt.close()
    
    # Save predictions for best depth
    best_clf = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)
    best_clf.fit(X_train, y_train)
    pd.DataFrame({'prediction': best_clf.predict(X_test)}).to_csv(
        os.path.join(output_folder, 'prediction_d_max_depth.csv'), 
        index=False
    )
    
    # Part d(ii): Vary ccp_alpha with scikit-learn
    alphas = [0.001, 0.01, 0.1, 0.2]
    train_accs, val_accs, test_accs = [], [], []
    best_alpha = None
    best_accuracy = 0
    
    print("\nPart d(ii): Varying ccp_alpha with scikit-learn")
    # First grow a full tree
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    
    for alpha in alphas:
        pruned_clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha)
        pruned_clf.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, pruned_clf.predict(X_train))
        val_acc = accuracy_score(y_valid, pruned_clf.predict(X_valid))
        test_acc = accuracy_score(y_test, pruned_clf.predict(X_test))
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        
        print(f"Alpha {alpha}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_alpha = alpha
    
    # Plot accuracy vs ccp_alpha
    plt.figure()
    plt.plot(alphas, train_accs, label='Train Accuracy')
    plt.plot(alphas, val_accs, label='Validation Accuracy')
    plt.plot(alphas, test_accs, label='Test Accuracy')
    plt.xlabel('CCP Alpha')
    plt.ylabel('Accuracy')
    plt.title('Scikit-learn Decision Tree: Accuracy vs CCP Alpha')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'part_d_ccp_alpha.png'))
    plt.close()
    
    # Save predictions for best alpha
    best_clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=best_alpha)
    best_clf.fit(X_train, y_train)
    pd.DataFrame({'prediction': best_clf.predict(X_test)}).to_csv(
        os.path.join(output_folder, 'prediction_d_ccp_alpha.csv'), 
        index=False
    )

def part_e(train_path, valid_path, test_path, output_folder):
    # Load and preprocess data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(train_path, valid_path, test_path)
    
    # Apply one-hot encoding (as in part b)
    X_train, X_valid, X_test, categorical_cols = one_hot_encode_data(X_train, X_valid, X_test)
    
    # Label encode remaining categorical columns (as in part d)
    remaining_categorical = X_train.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_categorical:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_valid[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_valid[col] = le.transform(X_valid[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 150, 250, 350],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [2, 4, 6, 8, 10]
    }
    
    best_oob = -1
    best_params = None
    best_clf = None
    
    # Perform grid search over parameters using OOB accuracy
    for params in ParameterGrid(param_grid):
        print(f"Training with params: {params}")
        clf = RandomForestClassifier(
            criterion='entropy',
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            min_samples_split=params['min_samples_split'],
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        oob_accuracy = clf.oob_score_
        print(f"OOB Accuracy: {oob_accuracy:.4f}")
        
        if oob_accuracy > best_oob:
            best_oob = oob_accuracy
            best_params = params
            best_clf = clf
            print("New best OOB accuracy found!")
    
    # After finding the best model
    print("\nBest Parameters:", best_params)
    print(f"Best OOB Accuracy: {best_oob:.4f}")
    
    # Calculate training, validation, and test accuracies
    train_pred = best_clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    val_pred = best_clf.predict(X_valid)
    val_acc = accuracy_score(y_valid, val_pred)
    
    test_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save predictions
    pd.DataFrame({'prediction': test_pred}).to_csv(
        os.path.join(output_folder, 'prediction_e.csv'), 
        index=False
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('validation_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    parser.add_argument('output_folder_path', type=str)
    parser.add_argument('question_part', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    if args.question_part == 'a':
        part_a(args.train_data_path, args.validation_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'b':
        part_b(args.train_data_path, args.validation_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'c':
        part_c(args.train_data_path, args.validation_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'd':
        part_d(args.train_data_path, args.validation_data_path, args.test_data_path, args.output_folder_path)
    elif args.question_part == 'e':
        part_e(args.train_data_path, args.validation_data_path, args.test_data_path, args.output_folder_path)

if __name__ == '__main__':
    main()