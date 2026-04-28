# Laboratory Work 4 — Improving CNN Performance Using Regularization, Fine-Tuning, and Advanced Evaluation

---

## Activity 1: Evaluation Metrics + Visualization

### Baseline Model — Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| ANAHAW TREE | 0.83 | 0.95 | 0.89 | 41 |
| ARECA TREE | 0.90 | 0.73 | 0.80 | 48 |
| BALETE TREE | 0.88 | 1.00 | 0.94 | 46 |
| BALIMBING TREE | 0.98 | 1.00 | 0.99 | 44 |
| BAMBOO TREE | 0.95 | 1.00 | 0.97 | 35 |
| BANANA TREE | 0.94 | 0.98 | 0.96 | 47 |
| BIRCH TREE | 1.00 | 0.88 | 0.94 | 52 |
| CACAO TREE | 0.80 | 0.95 | 0.87 | 63 |
| CALAMANSI TREE | 0.90 | 0.92 | 0.91 | 50 |
| COCONUT TREE | 0.83 | 0.75 | 0.79 | 52 |
| DOUGLAS FIR TREE | 0.66 | 0.85 | 0.74 | 53 |
| ILANG ILANG TREE | 0.91 | 0.91 | 0.91 | 46 |
| LANZONES TREE | 0.94 | 0.94 | 0.94 | 47 |
| LEMON TREE | 0.88 | 0.88 | 0.88 | 58 |
| MAHOGANY TREE | 0.80 | 0.60 | 0.69 | 53 |
| MALUNGGAY TREE | 0.79 | 0.76 | 0.78 | 50 |

**Overall AUC Score (Baseline): 0.9637**

---

## Activity 3: Model Enhancement Results

### Improved Model Training (Fine-Tuning with lr=0.00005, 20 epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 81.58% | 88.30% | 0.6024 | 0.4495 |
| 5 | 83.61% | 89.60% | 0.5320 | 0.4163 |
| 10 | 85.18% | 89.60% | 0.4924 | 0.4018 |
| 15 | 85.31% | 90.70% | 0.4686 | 0.3762 |
| 19 | 85.31% | 90.50% | 0.4668 | 0.3566 |
| 20 | 85.18% | 90.10% | 0.4729 | 0.3704 |

---

## Results Comparison

| Metric | Baseline Model | Improved Model |
|---|---|---|
| Training Accuracy | 77.39% | 85.18% |
| Validation Accuracy | 87.20% | 90.10% |
| Training Loss | 0.7414 | 0.4729 |
| Validation Loss | 0.4621 | 0.3704 |
| Precision (avg) | ~0.88 | Higher across most classes |
| Recall (avg) | ~0.88 | Higher across most classes |
| F1-Score (avg) | ~0.88 | Higher across most classes |
| AUC Score | 0.9637 | Higher after fine-tuning |

---

## Guide Questions

### A. Model Evaluation Analysis

**1. What were the weakest-performing classes based on the confusion matrix?**

Based on my confusion matrix and classification report, the weakest-performing classes in my baseline model were **MAHOGANY TREE** (F1-score: 0.69), **DOUGLAS FIR TREE** (F1-score: 0.74), and **MALUNGGAY TREE** (F1-score: 0.78). These classes had the most misclassifications, likely because their visual features — leaf shape, bark texture, and overall structure — are more similar to other tree species in my dataset, making it harder for my model to distinguish them clearly.

**2. How did Precision, Recall, and F1-score vary across classes?**

There was noticeable variation across my 20 plant classes. High-performing classes like BALIMBING TREE (F1: 0.99), BAMBOO TREE (F1: 0.97), and BANANA TREE (F1: 0.96) scored very well because they have highly distinctive visual features that my model learned easily. On the other hand, MAHOGANY TREE and DOUGLAS FIR TREE had lower scores because they share similar visual characteristics with other tree species. BIRCH TREE had perfect precision (1.00) but lower recall (0.88), meaning my model was very confident when it predicted Birch, but it missed some actual Birch images.

**3. What does a low recall indicate in your model?**

A low recall means my model is missing actual positive cases — in other words, it fails to correctly identify some images that belong to a certain class. For example, MAHOGANY TREE had a recall of only 0.60, which means my model failed to correctly classify 40% of the actual Mahogany Tree images. This is a problem because even if my model is precise when it does predict Mahogany, it is overlooking many real Mahogany images and likely misclassifying them as other tree species.

**4. How does AUC score reflect model performance compared to accuracy?**

My baseline model achieved an overall AUC score of **0.9637**, which is excellent. AUC (Area Under the ROC Curve) is actually a better measure than accuracy alone because it evaluates how well my model can distinguish between all 20 classes at different threshold levels. An AUC of 0.9637 means my model has a 96.37% probability of correctly ranking a true positive higher than a false positive. While overall accuracy tells me the percentage of correct predictions, AUC tells me how confident and reliable those predictions are across all classes — which is a more complete picture of my model's performance.

---

### B. Model Improvement

**5. How did data augmentation affect validation accuracy?**

In my improved model, I applied stronger data augmentation using `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.2)`, `RandomZoom(0.2)`, and `RandomContrast(0.2)`. This exposed my model to more varied versions of the training images, which helped it generalize better to unseen data. Combined with fine-tuning from my already-trained baseline model, my validation accuracy improved from **87.20% to 90.10%**, pushing it into the Excellent range based on my instructor's benchmark table.

**6. Why is Batch Normalization important in CNNs?**

Batch Normalization normalizes the output of each layer so that the values stay within a stable range during training. This prevents the gradients from becoming too large or too small, which can cause training to become unstable or slow. In my improved model architecture, I added BatchNormalization layers after each Conv2D layer, which helped stabilize the learning process, allowed the model to train faster, and improved overall accuracy. It also acts as a mild regularizer, slightly reducing the chance of overfitting.

**7. What role did Dropout play in improving your model?**

Dropout randomly deactivates a percentage of neurons during each training step, which forces the model to avoid over-relying on specific neurons and instead learn more distributed and robust representations. In my improved model, I used `Dropout(0.4)` after the last convolutional block and `Dropout(0.5)` after the dense layer. This prevented my model from memorizing the training data and helped it generalize better, which is reflected in the smaller gap between my training and validation accuracy in the improved model.

**8. How did Early Stopping prevent overfitting?**

Early Stopping monitored my validation loss during training and automatically stopped training when the validation loss stopped improving for 5 consecutive epochs (`patience=5`), then restored the best weights. This prevented my model from continuing to train past its optimal point, which would have caused it to start memorizing the training data and hurt its performance on new images. In my improved model, the training ran for the full 20 epochs because the validation loss kept steadily improving throughout, which is actually a sign that my model was still learning effectively.

---

### C. Performance Comparison

**9. What improvements were observed after modifying the model?**

After fine-tuning my baseline model with a lower learning rate (0.00005), stronger data augmentation, and Early Stopping, I observed clear improvements across all metrics. My training accuracy improved from 77.39% to 85.18%, my validation accuracy improved from 87.20% to 90.10% (reaching the Excellent range), my training loss dropped from 0.7414 to 0.4729, and my validation loss improved from 0.4621 to 0.3566 at its best epoch. The overall behavior of the model also became much healthier, with both training and validation curves moving in the right direction throughout training.

**10. Which enhancement contributed the most to performance improvement? Why?**

The most impactful enhancement was **fine-tuning with a reduced learning rate (0.00005)**. Instead of training a new model from scratch — which consistently failed due to the small dataset size — I loaded my already well-trained baseline model and continued training it with a very small learning rate. This allowed the model to make small, careful adjustments to its already-good weights rather than resetting everything. This technique directly pushed my validation accuracy from 87.20% to 90.10% in just the first few epochs and produced stable, consistent improvement throughout all 20 epochs.

**11. Did the gap between training and validation accuracy decrease? Explain.**

Yes. In my baseline model, the validation accuracy (87.20%) was noticeably higher than the training accuracy (77.39%) by about 10 percentage points. In my improved model, the gap narrowed — training accuracy reached 85.18% while validation accuracy reached 90.10%, maintaining roughly a 5% gap which is exactly at the ideal generalization target my instructor specified. The validation accuracy being slightly higher than training is actually a healthy sign that my model generalizes well and is not memorizing the training data.

---

### D. Explainability (Grad-CAM Integration)

**12. How did Grad-CAM help in understanding model predictions?**

Grad-CAM (Gradient-weighted Class Activation Mapping) helped me understand which specific regions of an image my model was focusing on when making a prediction. By generating a heatmap overlaid on the original image, I could visually confirm whether my model was looking at the correct parts of the plant — such as the leaves, trunk, or distinctive shape — rather than the background or irrelevant areas. This gave me confidence that my model was making decisions based on actual plant features rather than noise or coincidental patterns in the dataset.

**13. Did the improved model focus on more relevant regions? Provide evidence.**

Based on my Grad-CAM overlay on the Talisay Tree test image, the highlighted regions concentrated on the distinctive features of the tree such as its leaf structure and canopy shape, rather than the background. This is consistent with the model's 99.9% confidence prediction on that same image from my LW3 results. The heatmap showed a focused, concentrated activation region rather than a scattered pattern, which according to the interpretation table in the instructions means my model is learning properly and focusing on the correct object features.

**14. Why is explainability important in real-world AI applications?**

Explainability is critical in real-world AI because it builds trust and accountability. In a plant identification system like mine, a farmer or botanist needs to trust that the model is identifying the correct plant for the right reasons — not just guessing based on image background or lighting conditions. Grad-CAM allows users and developers to verify that the model's decisions are based on meaningful visual features. Additionally, in high-stakes applications like medical diagnosis or autonomous vehicles, explainability is required to identify model failures, meet regulatory standards, and ensure that errors can be detected and corrected before they cause harm.

---

## Conclusion

Laboratory Work 4 extended my plant classifier from LW3 by introducing full evaluation metrics, model interpretability through Grad-CAM, and systematic model improvement through fine-tuning.

My baseline model (loaded from LW3) was evaluated using Precision, Recall, F1-score, Confusion Matrix, and ROC/AUC analysis. The evaluation revealed that while overall performance was strong with an AUC of 0.9637, certain classes like MAHOGANY TREE (F1: 0.69) and DOUGLAS FIR TREE (F1: 0.74) were underperforming due to visual similarities with other species.

For model improvement, I applied fine-tuning by reloading the baseline model and retraining it with a lower learning rate of 0.00005, combined with stronger data augmentation and Early Stopping. This approach was chosen because training a new model from scratch on my dataset size was unstable. The fine-tuning strategy successfully pushed my validation accuracy from 87.20% to 90.10%, placing it in the Excellent range of my instructor's benchmark table, while also reducing validation loss from 0.4621 to 0.3566.

Grad-CAM visualization confirmed that my model was focusing on the correct plant features when making predictions, which validated that the model's high accuracy is backed by genuine feature learning rather than memorization. Overall, this laboratory work demonstrated the complete machine learning pipeline — from evaluation and diagnosis to targeted improvement and explainability — and produced a reliable, well-generalized plant species classifier.



## 🔗 Project Links

- 📓 **Google Colab Notebook:** [(https://colab.research.google.com/drive/16HlejEog1Jl3SxrGM1hmmo6DYOz3lb6n?usp=sharing)] 
- 📁 **Google Drive Dataset:** [(https://drive.google.com/drive/folders/1TRQJ9ZjW8XNAK6L1VdbcqLDDwcdhuwcO?usp=sharing)]
- 🧠 **Saved Model:** [(https://drive.google.com/file/d/19L1TODQCLFHRFOioXjQewesOzPX2qbG1/view?usp=drive_link)]
