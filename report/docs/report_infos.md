## CLI commands

```
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd_sw --model_s resnet8x4 -a 0 -b 0.8 --trial 1 --sw_alpha 1 --sw_tau 0.55
```

```
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd_sw --model_s wrn_40_1 -a 0 -b 0.8 --trial 1 --sw_alpha 1 --sw_tau 0.5
```

### Extract csv from the `logs` folder

    ```bash
    python3 report/extract_log.py --run_dir logs/kd/ --output report/logs/kd.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_rkd_r:1_a:0.0_b:1.0_1 --output report/logs/rks.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_r:1_a:0.0_b:0.8_1 --output report/logs/crd.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_sw_r:1_a:0.0_b:0.8_1 --output report/logs/crd_sw.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.55_1 --output report/logs/crd_sw:1.0_tau:0.55_1.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.5_1_4096 --output report/logs/crd_sw_k4096.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.5_1_64 --output report/logs/crd_sw_k64.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:resnet8x4_T:resnet32x4_cifar100_crd_sw_r:1_a:1.0_b:0.8_sw:1.0_tau:0.5_1 --output report/logs/crd_sw_a1.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.55_1 --output report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.55_1.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.6_1 --output report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.6_1.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:wrn_40_1_T:wrn_40_2_cifar100_crd_r:1_a:0.0_b:0.8_sw:1.0_tau:None_1 --output report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_r:1_a:0.0_b:0.8_sw:1.0_tau:None_1.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.4_1 --output report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.4_1.csv
    ```

    ```bash
    python3 report/extract_log.py --run_dir logs/S:wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.6_1 --output report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.6_1.csv
    ```

### plot training curves
    ```bash
    python3 report/plot_training_curves.py --csv report/logs/kd.csv report/logs/rkd.csv report/logs/crd.csv --out_dir report/graphs
    ```

    ```bash
    python3 report/plot_training_curves.py --csv report/logs/crd.csv report/logs/kd.csv report/logs/crd_sw:1.0_tau:0.5_1.csv --out_dir report/graphs
    ```

    ```bash
    python3 report/plot_training_curves.py --csv report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_r:1_a:0.0_b:0.8_sw:1.0_tau:None_1.csv report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.4_1.csv report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.6_1.csv --out_dir report/graphs
    ```

### report best metrics

    ```bash
    python3 report/report_best_metrics.py --csv report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_r:1_a:0.0_b:0.8_sw:1.0_tau:None_1.csv
    ```

## Metric values

Results for `report/logs/kd.csv`:
- Training Loss: 1.7305 at epoch 223
- Test Loss: 1.0781 at epoch 187
- Training Accuracy: 89.0480 at epoch 240
- Test Accuracy: 74.1900 at epoch 206
- Test Top-5 Accuracy: 93.4100 at epoch 236

---

Results for report/logs/rkd.csv:
- Training Loss: 0.7417 at epoch 232
- Test Loss: 0.9939 at epoch 183
- Training Accuracy: 93.6040 at epoch 237
- Test Accuracy: 72.7100 at epoch 193
- Test Top-5 Accuracy: 92.8900 at epoch 181

Test accuracy imporved from kd: -0.01994878015905114

---

Results for `report/logs/crd.csv`:
- Training Loss: 3.3963 at epoch 240
- Test Loss: 0.8618 at epoch 197
- Training Accuracy: 92.3980 at epoch 228
- Test Accuracy: 75.5800 at epoch 233
- Test Top-5 Accuracy: 94.0200 at epoch 195

Test accuracy imporved from kd: 0.01873567866289258


---

Results for report/logs/crd_sw:1.0_tau:0.05_1.csv (alpha=1.0, tau=0.05):
- Training Loss: 12.4528 at epoch 180
- Test Loss: 0.8753 at epoch 187
- Training Accuracy: 92.4600 at epoch 211
- Test Accuracy: 75.5500 at epoch 188
- Test Top-5 Accuracy: 94.1900 at epoch 203

---

Results for report/logs/crd_sw:1.0_1.csv (alpha=1.0, tau=0.07):
- Training Loss: 7.9158 at epoch 180
- Test Loss: 0.8671 at epoch 182
- Training Accuracy: 92.7960 at epoch 228
- Test Accuracy: 75.8800 at epoch 193
- Test Top-5 Accuracy: 94.1100 at epoch 183

---

Results for report/logs/crd_sw:1.0_tau:0.35_1.csv (alpha=1.0, tau=0.35):
- Training Loss: 4.1231 at epoch 211
- Test Loss: 0.8691 at epoch 221
- Training Accuracy: 92.6100 at epoch 231
- Test Accuracy: 75.8300 at epoch 191
- Test Top-5 Accuracy: 94.1200 at epoch 236

---

Results for report/logs/crd_sw:1.0_tau:0.45_1.csv (alpha=1.0, tau=0.45):
- Training Loss: 4.0954 at epoch 211
- Test Loss: 0.8655 at epoch 222
- Training Accuracy: 92.5220 at epoch 235
- Test Accuracy: 75.8700 at epoch 235
- Test Top-5 Accuracy: 94.4400 at epoch 224

---

Results for report/logs/crd_sw:1.0_tau:0.5_1.csv (alpha=1.0, tau=0.5):
- Training Loss: 4.0808 at epoch 212
- Test Loss: 0.8525 at epoch 225
- Training Accuracy: 92.3480 at epoch 222
- Test Accuracy: 76.0100 at epoch 233
- Test Top-5 Accuracy: 94.5400 at epoch 216

---

Results for report/logs/crd_sw:1.0_tau:0.55_1.csv (alpha=1.0, tau=0.55):
- Training Loss: 4.0963 at epoch 212
- Test Loss: 0.8702 at epoch 191
- Training Accuracy: 92.2280 at epoch 236
- Test Accuracy: 75.2200 at epoch 209
- Test Top-5 Accuracy: 94.2300 at epoch 193

---

Results for report/logs/crd_sw:1.0_tau:0.6_1.csv (alpha=1.0, tau=0.6):
- Training Loss: 4.1058 at epoch 212
- Test Loss: 0.8626 at epoch 191
- Training Accuracy: 92.2800 at epoch 232
- Test Accuracy: 75.3400 at epoch 224
- Test Top-5 Accuracy: 94.2300 at epoch 226

---

Results for report/logs/crd_sw:1.0_tau:1.0_1.csv (alpha=1.0, tau=1.0):
- Training Loss: 4.1299 at epoch 212
- Test Loss: 0.8654 at epoch 194
- Training Accuracy: 92.1460 at epoch 229
- Test Accuracy: 75.4100 at epoch 212
- Test Top-5 Accuracy: 94.1900 at epoch 239

---

Results for report/logs/crd_sw:0.5_1.csv (alpha=0.5, tau=0.07):
- Training Loss: 7.7616 at epoch 180
- Test Loss: 0.8698 at epoch 193
- Training Accuracy: 92.6880 at epoch 221
- Test Accuracy: 75.4800 at epoch 203
- Test Top-5 Accuracy: 94.0700 at epoch 199

---

Results for report/logs/crd_sw:1.0_tau:0.5_1_a1.csv:
- Training Loss: 5.7994 at epoch 211
- Test Loss: 0.9527 at epoch 185
- Training Accuracy: 93.9420 at epoch 237
- Test Accuracy: 75.9900 at epoch 225
- Test Top-5 Accuracy: 93.9400 at epoch 186

---

Results for report/logs/crd_sw_k64.csv:
- Training Loss: 1.2501 at epoch 211
- Test Loss: 0.9168 at epoch 188
- Training Accuracy: 94.6040 at epoch 228
- Test Accuracy: 74.6300 at epoch 186
- Test Top-5 Accuracy: 93.7500 at epoch 191

---

Results for report/logs/crd_sw_k256.csv:
- Training Loss: 1.6806 at epoch 211
- Test Loss: 0.8982 at epoch 183
- Training Accuracy: 94.3800 at epoch 225
- Test Accuracy: 75.0800 at epoch 222
- Test Top-5 Accuracy: 94.0300 at epoch 182

---

Results for report/logs/crd_sw_k1024.csv:
- Training Loss: 2.2658 at epoch 211
- Test Loss: 0.8772 at epoch 189
- Training Accuracy: 93.9800 at epoch 235
- Test Accuracy: 75.6300 at epoch 236
- Test Top-5 Accuracy: 94.2700 at epoch 193

---

Results for report/logs/crd_sw_k4096.csv:
- Training Loss: 3.0190 at epoch 211
- Test Loss: 0.8693 at epoch 184
- Training Accuracy: 93.2680 at epoch 219
- Test Accuracy: 75.7400 at epoch 203 (0.36%)
- Test Top-5 Accuracy: 93.9600 at epoch 213 (0.62%)

---

Results for report/logs/crd_sw:1.0_tau:0.5_1.csv (16384):
- Training Loss: 4.0808 at epoch 212
- Test Loss: 0.8525 at epoch 225
- Training Accuracy: 92.3480 at epoch 222
- Test Accuracy: 76.0100 at epoch 233
- Test Top-5 Accuracy: 94.5400 at epoch 216

---

Results for report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_r:1_a:0.0_b:0.8_sw:1.0_tau:None_1.csv:
- Training Loss: 3.8709 at epoch 239
- Test Loss: 0.9737 at epoch 181
- Training Accuracy: 91.9720 at epoch 239
- Test Accuracy: 74.7300 at epoch 192
- Test Top-5 Accuracy: 93.3500 at epoch 189

---

Results for report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.55_1.csv:
- Training Loss: 5.0194 at epoch 211
- Test Loss: 0.9848 at epoch 181
- Training Accuracy: 92.1360 at epoch 238
- Test Accuracy: 74.0500 at epoch 199
- Test Top-5 Accuracy: 93.3200 at epoch 184

---

Results for report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.6_1.csv:
- Training Loss: 4.9915 at epoch 211
- Test Loss: 0.9827 at epoch 183
- Training Accuracy: 92.3460 at epoch 236
- Test Accuracy: 74.5000 at epoch 221
- Test Top-5 Accuracy: 93.5200 at epoch 188

---

Results for report/logs/wrn_40_1_T:wrn_40_2_cifar100_crd_sw_r:1_a:0.0_b:0.8_sw:1.0_tau:0.4_1.csv:
- Training Loss: 5.1825 at epoch 211
- Test Loss: 0.9814 at epoch 157
- Training Accuracy: 92.3280 at epoch 229
- Test Accuracy: 74.8600 at epoch 224
- Test Top-5 Accuracy: 93.3500 at epoch 190

## Analysis of Results

### Preliminary Results

We report training loss, test loss, training accuracy, and test accuracy curves for KD, RKD, and CRD over 240 epochs on CIFAR-100, along with final performance metrics. As shown in Figure~\ref{fig:kd-curves}, all methods converge successfully with visible improvements after the scheduled learning-rate decay steps. CRD achieves the lowest final training and test losses and the highest accuracy, while RKD converges similarly in training accuracy but reaches a lower test accuracy. Table~\ref{tab:kd-results} summarizes the final results, confirming that CRD outperforms the KD baseline while RKD underperforms.

![Test Accuracy vs Epoch](report/graphs/test_accuracy_vs_epoch.png)

### Evaluation

We evaluated three knowledge-distillation strategies on CIFAR-100: the vanilla Knowledge Distillation baseline (KD), Relational Knowledge Distillation (RKD), and Contrastive Representation Distillation (CRD). We expected CRD to outperform KD due to its stronger contrastive objective, and anticipated RKD to provide moderate improvements by leveraging relational structure between samples.

The experimental results largely align with these expectations. CRD achieves the highest test accuracy (75.58\%) and top-5 accuracy (94.02\%), outperforming KD by +1.39\% in top-1 accuracy while also attaining the lowest test loss. This improvement is reflected in the training dynamics: CRD shows consistently lower test loss and stronger generalization signals after the scheduled learning-rate decay phases, suggesting more discriminative feature representations. KD performs competitively, serving as a strong baseline with stable convergence behavior and solid accuracy (74.19\%). 

In contrast, RKD did not fully meet our expectations. Although it has lower training loss than KD and even achieving a similar training accuracy level to CRD, its performance deteriorates noticeably on the test set. Despite appearing to learn effectively during training, RKD suffers a −1.49\% drop in top-1 accuracy compared to KD and exhibits a higher test loss, indicating weaker generalization. This suggests that RKD's pairwise relational constraints, though beneficial for capturing local sample structure, may not provide sufficiently rich or globally consistent supervisory signals to guide the student toward transferable feature representations. These data findings align with prior literature.

In our experiments, CRD improves student accuracy and stability, whereas RKD struggles to generalize despite strong training performance. This suggests that contrastive supervision provides a more transferable signal in this setting, while relational constraints may require further tuning to be effective.
