import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def evaluate(model, test_loader, device):

    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_predictions, all_labels, all_probs

def compute_metrics(all_predictions, all_labels):

    class_names = ['CN', 'MCI', 'Dementia']

    # Classification report
    print("Classification report:")
    print(metrics.classification_report(all_labels, all_predictions, target_names=class_names))

    # Overall accuracy
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    print(f"Overall accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = metrics.confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.savefig('../outputs/figures/confusion_matrix.png', dpi=150)
    plt.show()
    
    return accuracy, cm

def compute_auc(all_labels, all_probs):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    auc = metrics.roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f"Macro AUC-ROC: {auc:.4f}")
    
    return auc

def bootstrap_confidence_interval(all_labels, all_predictions, n_bootstrap=1000, ci=0.95):
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    bootstrap_accuracies = []
    n = len(all_labels)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, n, replace=True)
        boot_labels = all_labels[indices]
        boot_preds = all_predictions[indices]
        acc = metrics.accuracy_score(boot_labels, boot_preds)
        bootstrap_accuracies.append(acc)
    
    lower = np.percentile(bootstrap_accuracies, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_accuracies, (1 + ci) / 2 * 100)
    
    print(f"Accuracy: {metrics.accuracy_score(all_labels, all_predictions):.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")
    
    return lower, upper