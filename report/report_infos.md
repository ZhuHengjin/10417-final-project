## CLI commands

- Extract csv from the `logs` folder

    ```bash
    python3 report/extract_log.py --run_dir logs/kd/ --output report/logs/kd.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_rkd_r:1_a:0.0_b:1.0_1 --output report/logs/rks.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_r:1_a:0.0_b:0.8_1 --output report/logs/crd.csv
    ```

- plot training curves
    ```bash
    python3 report/plot_training_curves.py --csv report/logs/kd.csv report/logs/rkd.csv report/logs/crd.csv --out_dir report/graphs
    ```

- report best metrics

    ```bash
    python3 report/report_best_metrics.py --csv report/logs/crd.csv
    ```

## Metric values

Results for `report/logs/kd.csv`:
- Training Loss: 1.7305 at epoch 223
- Test Loss: 1.0781 at epoch 187
- Training Accuracy: 89.0480 at epoch 240
- Test Accuracy: 74.1900 at epoch 206
- Test Top-5 Accuracy: 93.4100 at epoch 236

---

Results for `report/logs/crd.csv`:
- Training Loss: 3.3963 at epoch 240
- Test Loss: 0.8618 at epoch 197
- Training Accuracy: 92.3980 at epoch 228
- Test Accuracy: 75.5800 at epoch 233
- Test Top-5 Accuracy: 94.0200 at epoch 195

Test accuracy imporved from kd: 0.01873567866289258

---

Results for report/logs/rkd.csv:
- Training Loss: 0.7417 at epoch 232
- Test Loss: 0.9939 at epoch 183
- Training Accuracy: 93.6040 at epoch 237
- Test Accuracy: 72.7100 at epoch 193
- Test Top-5 Accuracy: 92.8900 at epoch 181

Test accuracy imporved from kd: -0.01994878015905114

## Analysis of Results

### Preliminary Results

We report training loss, test loss, training accuracy, and test accuracy curves for KD, RKD, and CRD over 240 epochs on CIFAR-100, along with final performance metrics. As shown in Figure~\ref{fig:kd-curves}, all methods converge successfully with visible improvements after the scheduled learning-rate decay steps. CRD achieves the lowest final training and test losses and the highest accuracy, while RKD converges similarly in training accuracy but reaches a lower test accuracy. Table~\ref{tab:kd-results} summarizes the final results, confirming that CRD outperforms the KD baseline while RKD underperforms.

![Test Accuracy vs Epoch](report/graphs/test_accuracy_vs_epoch.png)

### Evaluation

We evaluated three knowledge-distillation strategies on CIFAR-100: the vanilla Knowledge Distillation baseline (KD), Relational Knowledge Distillation (RKD), and Contrastive Representation Distillation (CRD). We expected CRD to outperform KD due to its stronger contrastive objective, and anticipated RKD to provide moderate improvements by leveraging relational structure between samples.

The experimental results largely align with these expectations. CRD achieves the highest test accuracy (75.58\%) and top-5 accuracy (94.02\%), outperforming KD by +1.39\% in top-1 accuracy while also attaining the lowest test loss. This improvement is reflected in the training dynamics: CRD shows consistently lower test loss and stronger generalization signals after the scheduled learning-rate decay phases, suggesting more discriminative feature representations. KD performs competitively, serving as a strong baseline with stable convergence behavior and solid accuracy (74.19\%). 

In contrast, RKD did not fully meet our expectations. Although it has lower training loss than KD and even achieving a similar training accuracy level to CRD, its performance deteriorates noticeably on the test set. Despite appearing to learn effectively during training, RKD suffers a −1.49\% drop in top-1 accuracy compared to KD and exhibits a higher test loss, indicating weaker generalization. This suggests that RKD's pairwise relational constraints, though beneficial for capturing local sample structure, may not provide sufficiently rich or globally consistent supervisory signals to guide the student toward transferable feature representations. These data findings align with prior literature.

In our experiments, CRD improves student accuracy and stability, whereas RKD struggles to generalize despite strong training performance. This suggests that contrastive supervision provides a more transferable signal in this setting, while relational constraints may require further tuning to be effective.
