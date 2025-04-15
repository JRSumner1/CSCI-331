import sys # For command-line arguments and system exit
import math # For mathematical functions
import pickle # For saving and loading models
import collections # For counting and default dictionaries
import random # For shuffling the data in the testing paramters function

# Class representing a node in the decision tree
class DecisionTreeNode:
    __slots__ = ("attribute", "true", "false")  # Optimize memory usage
    def __init__(self, attribute, true=None, false=None):
        self.attribute = attribute   # Can be either a feature index or a label if it's a leaf node
        self.true = true             # Left child (True branch)
        self.false = false           # Right child (False branch)

# Class representing a node in the AdaBoost ensemble
class AdaBoostNode:
    __slots__ = ("h", "z")  # Optimize memory usage
    def __init__(self, h, z):
        self.h = h    # Weak learner (decision stump)
        self.z = z    # Weight (alpha) of the weak learner

# Function to Calculate the entropy of a dataset
def entropy(labels, weights=None):
    entropy = 0.0
    total = len(labels)
    total_weight = sum(weights) if weights is not None else total
    label_weight = collections.defaultdict(float)

    # Calculate entropy without weights
    if weights is None:
        # Get the count for all labels
        counts = collections.Counter(labels)

        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
    # Calculate entropy with weights
    else:
        # Collect weights for each label
        for label, weight in zip(labels, weights):
            label_weight[label] += weight

        for w in label_weight.values():
            p = w / total_weight
            if p > 0:
                entropy -= p * math.log2(p)

    return entropy

# Function to calculate information gain of a split on a given feature
def infoGain(data, feature_index, weights=None):
    # Get labels from data
    labels = [label for label, _ in data]

    # Calculate initial entropy
    if weights is None:
        initial_entropy = entropy(labels)
    else:
        initial_entropy = entropy(labels, weights)

    # Split data into subsets based on the feature
    subset_true = []     # Data where feature is True
    subset_false = []    # Data where feature is False
    weights_true = []    # Corresponding weights
    weights_false = []

    for i, (label, example) in enumerate(data):
        if example[feature_index]:
            subset_true.append((label, example))
            if weights is not None:
                weights_true.append(weights[i])
        else:
            subset_false.append((label, example))
            if weights is not None:
                weights_false.append(weights[i])

    # Get labels from subsets
    labels_true = [label for label, _ in subset_true]
    labels_false = [label for label, _ in subset_false]

    # Calculate entropies and weights of subsets
    if weights is None:
        total = len(labels)
        ent_true = entropy(labels_true)
        ent_false = entropy(labels_false)
        weight_true = len(labels_true) / total
        weight_false = len(labels_false) / total
    else:
        total_weight = sum(weights)
        ent_true = entropy(labels_true, weights_true)
        ent_false = entropy(labels_false, weights_false)
        weight_true = sum(weights_true) / total_weight
        weight_false = sum(weights_false) / total_weight

    # Calculate new entropy after the split
    new_entropy = weight_true * ent_true + weight_false * ent_false
    gain = initial_entropy - new_entropy

    return gain

# Function to calculate the majority label in the data, weighted if weights are provided
def weightedMajorityLabel(data, weights=None):
    label_weight = collections.defaultdict(float)

    if weights is None:
        # Count occurrences of each label
        for label, _ in data:
            label_weight[label] += 1
    else:
        # Collect weights for each label
        for (label, _), weight in zip(data, weights):
            label_weight[label] += weight

    # Select the label with the highest total weight/count
    majority_label = max(label_weight.items(), key=lambda item: item[1])[0]

    return majority_label

# Function to train a decision tree
def trainDecisionTree(data, features, max_depth, weights=None):
    # Get labels from data
    labels = [label for label, _ in data]
    # Determine the majority label (could be used as a fallback)
    majority_label = weightedMajorityLabel(data, weights)

    # Base cases for recursion
    if max_depth == 0:
        # If maximum depth reached, return a leaf node with the majority label
        return DecisionTreeNode(attribute=majority_label)
    elif len(data) == 0:
        # If no data, return a leaf node with the majority label
        return DecisionTreeNode(attribute=majority_label)
    elif all(label == labels[0] for label in labels):
        # If all labels are the same, return a leaf node with that label
        return DecisionTreeNode(attribute=labels[0])

    # Initialize variables to track the best feature to split on
    best_gain = -float('inf')
    best_feature_index = None
    num_features = len(features)
    # Loop over all features to find the one with the highest information gain
    for i in range(num_features):
        gain = infoGain(data, i, weights)
        if gain > best_gain:
            best_gain = gain
            best_feature_index = i

    # If no gain, return a leaf node with the majority label
    if best_feature_index is None:
        return DecisionTreeNode(attribute=majority_label)

    # Split data into subsets based on the best feature
    true_branch_data = []
    false_branch_data = []
    true_branch_weights = []
    false_branch_weights = []

    for i, (label, example) in enumerate(data):
        if example[best_feature_index]:
            true_branch_data.append((label, example))
            if weights is not None:
                true_branch_weights.append(weights[i])
        else:
            false_branch_data.append((label, example))
            if weights is not None:
                false_branch_weights.append(weights[i])

    # Recursively build the true and false branches
    true_branch = trainDecisionTree(true_branch_data, features, max_depth=max_depth-1,
                                    weights=true_branch_weights if weights is not None else None)
    false_branch = trainDecisionTree(false_branch_data, features, max_depth=max_depth-1,
                                     weights=false_branch_weights if weights is not None else None)

    # Return a decision node with the best feature and branches
    return DecisionTreeNode(attribute=best_feature_index, true=true_branch, false=false_branch)

# Function to train an AdaBoost ensemble
def trainAdaBoost(data, features, K):
    N = len(data)
    # Initialize weights uniformly
    weights = [1 / N] * N
    ensemble = []
    for K in range(K):
        # Train a decision stump (max_depth=1)
        stump = trainDecisionTree(data, features, max_depth=1, weights=weights)
        predictions = []
        # Make predictions on the data
        for label, example in data:
            prediction = predictDecisionTree(example, stump)
            predictions.append(prediction)
        # Calculate the weighted error
        error = sum(weights[i] for i in range(N) if predictions[i] != data[i][0])
        # If error is zero or too high, break
        if error == 0 or error >= 0.5:
            break
        # Calculate alpha
        alpha = 0.5 * math.log((1 - error) / error)
        # Update weights
        for i in range(N):
            if predictions[i] == data[i][0]:
                weights[i] *= math.exp(-alpha)
            else:
                weights[i] *= math.exp(alpha)
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        # Add the stump and its weight to the ensemble
        ensemble.append(AdaBoostNode(h=stump, z=alpha))
    return ensemble

# Function to predict the label of an example using a decision tree
def predictDecisionTree(example, node):
    if node.true is None and node.false is None:
        # If leaf node, return the label
        return node.attribute
    if example[node.attribute]:
        # If feature is True, go to the true branch
        return predictDecisionTree(example, node.true)
    else:
        # If feature is False, go to the false branch
        return predictDecisionTree(example, node.false)

# Function to predict the label of an example using an AdaBoost ensemble
def predictAdaBoost(example, ensemble):
    total = 0.0
    for node in ensemble:
        # Predict using each weak learner
        prediction = predictDecisionTree(example, node.h)
        # Convert prediction to +1 or -1
        if prediction == 'en':
            ht = 1
        else:
            ht = -1
        # Accumulate weighted predictions
        total += node.z * ht
    # Return the final prediction based on the sign of the total
    return 'en' if total > 0 else 'nl'

def calculateAccuracy(data, model, model_type):
    correct = 0
    total = len(data)
    for label, example in data:
        if model_type == 'dt':
            prediction = predictDecisionTree(example, model)
        else:
            prediction = predictAdaBoost(example, model)
        if prediction == label:
            correct += 1
    return correct / total

def testParameters(data, features, learning_type):
    random.shuffle(data)
    split_index = int(0.8 * len(data))
    training_data = data[:split_index]
    validation_data = data[split_index:]
    values = list(range(1,11))
    best_accuracy = 0.0
    best_model = None
    best_value = None

    if learning_type == 'dt':
        for max_depth in values:
            model = trainDecisionTree(training_data, features, max_depth=max_depth)
            accuracy = calculateAccuracy(validation_data, model, 'dt')
            print(f"max_depth: {max_depth}, Accuracy: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_value = max_depth
        print(f"Best max_depth: {best_value}, Best Accuracy: {best_accuracy:.4f}")
        return best_model
    else:
        for K in values:
            model = trainAdaBoost(training_data, features, K=K)
            accuracy = calculateAccuracy(validation_data, model, 'ada')
            print(f"K: {K}, Accuracy: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_value = K
        print(f"Best K: {best_value}, Best Accuracy: {best_accuracy:.4f}")
        return best_model

# Main function to handle training and prediction modes
def main():
    # Determine what mode is being passed
    mode = sys.argv[1]
    if mode != 'train' and mode != 'predict':
        print("Invalid mode. Use 'train' or 'predict'.")
        sys.exit(1)

    # Pass files from command-line based on the mode option
    if mode == 'train':
        # For training: examples_path, features_path, hypothesis_out, learning_type
        examples_path, features_path, hypothesis_out, learning_type = sys.argv[2:]
    else:
        # For prediction: examples_path, features_path, hypothesis_path
        examples_path, features_path, hypothesis_path = sys.argv[2:]

    # Create the features list for English and Dutch
    with open(features_path, 'r', encoding='utf-8') as file:
        features = [line.strip() for line in file]
    
    # Check if no features were found
    if not features:
        print("No valid features found.")
        sys.exit(1)

    # Train mode
    if mode == 'train':
        # Import and set training data
        training_data = []
        with open(examples_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Check if the label identifier is in the line
                if '|' not in line:
                    continue
                # Set label and the language string
                label, text = line.strip().split('|', 1)
                text = text.lower()
                # Create feature vector: 1 if feature is in text, else 0
                feature_vector = [1 if feature in text else 0 for feature in features]
                # Store in the training data list
                training_data.append((label.strip(), feature_vector))

        # Check if no data was found
        if not training_data:
            print("No valid training data found.")
            sys.exit(1)

        # Test to see which max_depth and K value are the most accurate (for testing purposes only)
        # model = testParameters(training_data, features, learning_type)

        # Learning type for decision tree
        if learning_type == 'dt':
            # Train a decision tree
            root = trainDecisionTree(training_data, features, max_depth=6)
            with open(hypothesis_out, 'wb') as file:
                pickle.dump(root, file)
        # Learning type for AdaBoost
        elif learning_type == 'ada':
            # Train an AdaBoost ensemble
            ensemble = trainAdaBoost(training_data, features, K=9)
            with open(hypothesis_out, 'wb') as file:
                pickle.dump(ensemble, file)
        # If learning type was not properly passed
        else:
            print("Invalid learning type. Use 'dt' for decision tree or 'ada' for AdaBoost.")
            sys.exit(1)
    # Predict mode
    else:
        # Import and set prediction data
        prediction_data = []
        with open(examples_path, 'r', encoding='utf-8') as file:
            for line in file:
                text = line.strip().lower()
                # Create feature vector
                feature_vector = [1 if feature in text else 0 for feature in features]
                prediction_data.append(feature_vector)

        # Check if no prediction data were found
        if not prediction_data:
            print("No valid prediction data.")
            sys.exit(1)

        # Open the model and read the file in binary
        with open(hypothesis_path, 'rb') as file:
            model = pickle.load(file)

        # Make predictions for each example
        for prediction in prediction_data:
            try:
                # Check if model was generated using decision tree
                if isinstance(model, DecisionTreeNode):
                    # Predict using decision tree
                    print(predictDecisionTree(prediction, model))
                # Check if model was generated using AdaBoost
                elif isinstance(model, list) and all(isinstance(node, AdaBoostNode) for node in model):
                    # Predict using AdaBoost
                    print(predictAdaBoost(prediction, model))
                # If model format is invalid
                else:
                    print("Invalid model format")
                    sys.exit(1)
            except ValueError as e:
                print(str(e))
                sys.exit(1)

if __name__ == "__main__":
    main()
