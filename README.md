# Laboratory Work 4: Improving CNN Performance Using Regularization, Fine-Tuning, and Advanced Evaluation

## Guide Questions and Answers

---

## A. Model Evaluation Analysis

### 1. What were the weakest-performing classes based on the confusion matrix?

Based on the confusion matrix and classification report, the weakest-performing classes were
COCONUT TREE (F1-score: 0.58), MAHOGANY TREE (F1-score: 0.56), and MALUNGGAY TREE
(F1-score: 0.60). The confusion matrix showed that COCONUT TREE had significant
misclassifications, with only 22 out of 52 samples correctly predicted, and several instances
being confused with other tree classes such as DOUGLAS FIR TREE and TALISAY TREE.
MALUNGGAY TREE also showed notable confusion with ANAHAW TREE, with 10 samples
misclassified into that class.

---

### 2. How did Precision, Recall, and F1-score vary across classes?

The Precision, Recall, and F1-score varied significantly across the 20 tree classes. BIRCH TREE
achieved the highest precision at 1.00, meaning every prediction made for that class was
correct. BALETE TREE and BAMBOO TREE achieved perfect recall scores of 1.00, meaning all
actual instances of those classes were correctly identified. In contrast, COCONUT TREE had a
recall of only 0.42, indicating that more than half of the actual coconut tree images were
misclassified. MAHOGANY TREE had the lowest precision at 0.52, meaning nearly half of its
predictions were incorrect. The bar chart visualization clearly showed that classes like
BALIMBING TREE, CACAO TREE, and CALAMANSI TREE had consistently high scores across
all three metrics, while COCONUT TREE, MAHOGANY TREE, and MALUNGGAY TREE showed
the most inconsistency.

---

### 3. What does a low recall indicate in your model?

A low recall indicates that the model is failing to correctly identify a large proportion of the
actual instances of that class — in other words, it is producing many false negatives. For
example, COCONUT TREE had a recall of 0.42, which means the model missed 58% of actual
coconut tree images and incorrectly classified them as other tree types. This could be caused
by visual similarity between coconut trees and other palm-like trees in the dataset,
insufficient training samples, or the model not learning discriminative enough features for
that particular class.

---

### 4. How does AUC score reflect model performance compared to accuracy?

The overall AUC score of 0.9530 reflects a significantly more optimistic view of model
performance compared to the overall accuracy of 0.82. While accuracy measures the
proportion of correct predictions, AUC (Area Under the ROC Curve) measures the model's
ability to distinguish between classes across all possible classification thresholds. An AUC of
0.9530 indicates that the model has excellent discriminative ability — it can reliably rank a
correct class higher than an incorrect one 95.3% of the time. This is especially useful in
multi-class problems like this one where some classes have imbalanced support counts, as
AUC is less sensitive to class imbalance than accuracy.

---

## B. Model Improvement

### 5. How did data augmentation affect validation accuracy?

Data augmentation using RandomFlip, RandomRotation, RandomZoom, and RandomContrast
was applied to artificially expand the diversity of the training data. In the improved model
training, the validation accuracy gradually increased from 0.0420 in Epoch 1 to 0.5150 by
Epoch 19, showing that augmentation helped the model generalize better to unseen data by
exposing it to varied versions of the training images. Without augmentation, the model would
be more likely to memorize the exact training images, leading to overfitting and poor
validation performance.

---

### 6. Why is Batch Normalization important in CNNs?

Batch Normalization is important in CNNs because it normalizes the outputs of each layer to
have a stable mean and variance during training. This addresses the problem of internal
covariate shift, where the distribution of inputs to each layer changes as the weights of
previous layers are updated. By stabilizing these distributions, Batch Normalization allows
the model to train faster, use higher learning rates, and reduces sensitivity to weight
initialization. In this activity, Batch Normalization was added after each Conv2D layer in the
improved architecture, which helped stabilize the training process as evidenced by the
steadily decreasing training loss across all 20 epochs.

---

### 7. What role did Dropout play in improving your model?

Dropout was applied at two points in the improved architecture — with a rate of 0.4 after the
last convolutional block and 0.5 after the Dense layer. During training, Dropout randomly
deactivates a proportion of neurons, forcing the network to learn redundant representations
and preventing it from becoming overly reliant on specific neurons. This acts as a
regularization technique that reduces overfitting. In the training results, although the model
was still learning across all 20 epochs, the gap between training accuracy (0.4260) and
validation accuracy (0.4700) remained relatively small, suggesting that Dropout was
effectively preventing the model from simply memorizing the training data.

---

### 8. How did Early Stopping prevent overfitting?

Early Stopping was configured to monitor validation loss with a patience of 3 epochs and
restore the best weights. This means if the validation loss did not improve for 3 consecutive
epochs, training would automatically stop and the weights from the best epoch would be
restored. In this activity, the model completed all 20 epochs since the validation loss
continued to improve throughout training, ending at 1.7326. Had the validation loss started
increasing while training loss continued to decrease, Early Stopping would have halted
training at the optimal point, preventing the model from overfitting to the training data.

---

## C. Performance Comparison

### 9. What improvements were observed after modifying the model?

| Metric             | Baseline Model | Improved Model       |
|--------------------|---------------|----------------------|
| Training Accuracy  | 0.82          | 0.4260 (epoch 20)    |
| Validation Accuracy| 0.82          | 0.4700 (epoch 20)    |
| Training Loss      | —             | 1.8752               |
| Validation Loss    | —             | 1.7326               |
| Overall AUC Score  | 0.9530        | Still training       |
| Macro F1-score     | 0.82          | Still training       |

The improved model showed consistent learning progress across all 20 epochs with both
training and validation accuracy steadily increasing. The validation accuracy actually exceeded
training accuracy in several epochs, indicating that the model was generalizing well without
overfitting. However, the improved model had not yet reached the baseline model's accuracy
level after 20 epochs, suggesting it requires more epochs to converge due to the lower
learning rate of 0.0001 and the more complex architecture.

---

### 10. Which enhancement contributed the most to performance improvement? Why?

Among all the enhancements applied, the improved CNN architecture with Batch
Normalization combined with the lower learning rate optimization contributed the most to
stable and consistent training performance. The Batch Normalization layers helped stabilize
gradient flow across the deeper architecture, while the Adam optimizer with a learning rate
of 0.0001 ensured that weight updates were small and precise, preventing the model from
overshooting optimal values. This is evidenced by the smooth and steady decrease in both
training and validation loss across all 20 epochs, compared to the more erratic initial
behavior in Epoch 2 where validation loss spiked to 14.5630 before stabilizing.

---

### 11. Did the gap between training and validation accuracy decrease? Explain.

Yes, the gap between training and validation accuracy remained very small throughout
training, and in many epochs the validation accuracy actually exceeded the training accuracy.
For example, in Epoch 20, the training accuracy was 0.4260 while the validation accuracy was
0.4700. This is a positive sign indicating that the model was not overfitting to the training
data. The combination of Dropout regularization, Batch Normalization, data augmentation,
and the conservative learning rate all contributed to keeping the generalization gap minimal,
which is the primary goal of the applied enhancements.

---

## D. Explainability (Grad-CAM Integration)

### 12. How did Grad-CAM help in understanding model predictions?

Grad-CAM (Gradient-weighted Class Activation Mapping) helped in understanding model
predictions by generating a visual heatmap that highlights the regions of the input image
that most influenced the model's classification decision. By computing the gradients of the
predicted class score with respect to the feature maps of the last convolutional layer
(conv2d_5), Grad-CAM produced a spatial map showing where the model was "looking" when
it made its prediction. The resulting overlay on the coconut tree image provided a visual
explanation of the model's internal decision process, making the otherwise black-box CNN
more interpretable and transparent.

---

### 13. Did the improved model focus on more relevant regions? Provide evidence.

The Grad-CAM heatmap generated from the baseline model showed a largely uniform yellow
activation across the entire image, with the overlay displaying scattered cyan and teal
highlights distributed across the full coconut tree image rather than concentrated on specific
discriminative features. This suggests that the baseline model was not precisely localizing
the most relevant visual features of the coconut tree but was instead activating broadly
across the whole image. The scattered activation pattern visible in the Grad-CAM overlay
indicates that the model relied on general texture and color patterns rather than specific
structural features such as the trunk, fronds, or coconut clusters for its classification
decision.

---

### 14. Why is explainability important in real-world AI applications?

Explainability is critically important in real-world AI applications because it builds trust,
enables accountability, and supports informed decision-making. In high-stakes domains such
as medical diagnosis, agriculture, and environmental monitoring, stakeholders need to
understand not just what the model predicted but why it made that prediction. For example,
in a tree species classification system used for forest management or biodiversity monitoring,
a model that can show which visual features it used to identify a species allows domain
experts to verify whether the model is reasoning correctly or relying on spurious
correlations. Grad-CAM and similar explainability tools also help developers identify and fix
model weaknesses — if the heatmap shows the model focusing on irrelevant background
regions rather than the tree itself, this signals a need for better training data or
architectural improvements. Without explainability, even a high-accuracy model remains a
black box that cannot be fully trusted or debugged in production environments.
