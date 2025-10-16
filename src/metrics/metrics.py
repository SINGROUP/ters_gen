import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class Metrics:

    def __init__(self, model, data, config):
        """
        Initialize the Metrics class with the model, data, and configuration.
        """
        self.model = model
        self.data = data
        self.config = config

    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy.
        """
        return np.mean(y_true == y_pred)

    def compute_precision(self, y_true, y_pred):
        """
        Compute precision.
        """
        return precision_score(y_true.flatten(), y_pred.flatten(), average='binary')

    def compute_recall(self, y_true, y_pred):
        """
        Compute recall.
        """
        return recall_score(y_true.flatten(), y_pred.flatten(), average='binary')

    def compute_f1_score(self, y_true, y_pred):
        """
        Compute F1-score.
        """
        return f1_score(y_true.flatten(), y_pred.flatten(), average='binary')

    def compute_iou(self, y_true, y_pred):
        """
        Compute Intersection over Union (IoU).
        """
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / union if union != 0 else 0

    def compute_dice_coefficient(self, y_true, y_pred):
        """
        Compute Dice coefficient.
        """
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2 * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 0

    def evaluate(self):
        """
        Evaluate the model on the provided data using all metrics.
        """
        y_true = self.data['ground_truth']
        #y_pred = self.model.predict(self.data['input'])
        y_pred = self.data['pred']

        metrics = {
            "Accuracy": self.compute_accuracy(y_true, y_pred),
            "Precision": self.compute_precision(y_true, y_pred),
            "Recall": self.compute_recall(y_true, y_pred),
            "F1 Score": self.compute_f1_score(y_true, y_pred),
            "IoU": self.compute_iou(y_true, y_pred),
            "Dice Coefficient": self.compute_dice_coefficient(y_true, y_pred),
        }

        return metrics
    


'''

    # Assuming you have a trained model and test data
model = ...  # Your segmentation model
test_data = {
    "input": X_test,  # Test inputs
    "ground_truth": y_test  # Ground truth labels
}
config = {}  # Any additional configuration

metrics = Metrics(model, test_data, config)
results = metrics.evaluate()

print("Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


    '''