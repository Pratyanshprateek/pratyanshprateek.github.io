# Medical Image Classification with ViT

## Internship Notes

This file is a practical walkthrough of the full project:

- what the training notebook does
- why some Mac-specific changes were needed
- how the prediction script works
- what warnings/errors we saw and what they actually meant
- common viva / internship Q&A

Style note:
Thoda Hinglish rakha hai so that it feels readable, not like dry documentation.

---

## 1. Project Goal

Humara project chest X-ray images ko classify karta hai into:

- `NORMAL`
- `PNEUMONIA`

Iske liye humne pretrained Vision Transformer model use kiya:

- `google/vit-base-patch16-224`

Simple language mein:
Humne ek already-trained image model liya, usko chest X-ray task ke liye fine-tune kiya, phir saved model se single image prediction bhi add ki.

---

## 2. Project Files

- [medical_image_classification_vit.ipynb](https://github.com/Pratyanshprateek/vision_transformer/blob/main/medical_image_classification_vit.ipynb)
Main training notebook. Isme data loading, balancing, training, evaluation, saving, visualization sab hai.

- [predict_single_xray.py](https://github.com/Pratyanshprateek/vision_transformer/blob/main/predict_single_xray.py)
Ek standalone script jo trained checkpoint use karke single X-ray predict karta hai.

- [README.md](https://github.com/Pratyanshprateek/vision_transformer/blob/main/README.md)
Project usage guide.

- outputs/vit_chest_xray_classifier.pt
Saved trained model checkpoint.

- outputs/training_history.csv
Training metrics history.

---

## 3. Dataset Structure

Expected dataset layout:

```text
./chest_xray/
  train/NORMAL
  train/PNEUMONIA
  val/NORMAL
  val/PNEUMONIA
  test/NORMAL
  test/PNEUMONIA
```

Project mein Kaggle Chest X-Ray dataset use hua.

Important baat:
Original dataset imbalanced hai. `PNEUMONIA` images zyada hain, `NORMAL` kam hain. Isi wajah se humne direct full dataset use nahi kiya; pehle balanced subset banaya.

---

## 4. Notebook Ka Full Flow

### 4.1 Imports and Setup

Notebook sabse pehle:

- required libraries import karta hai
- random seed set karta hai
- device choose karta hai
- batch size, epochs, learning rate jaise hyperparameters define karta hai

Mac support ke liye important settings:

- `FORCE_CPU = True`
- `TARGET_TOTAL_IMAGES = 3000` at the moment
- CPU runs ke liye smaller batch size
- `NUM_WORKERS = 0` for non-CUDA runs

Why?
Mac par Jupyter + multiprocessing + MPS combo unstable ho raha tha. Isliye safe local setup rakha gaya.

---

### 4.2 Device Selection

Code device choose karta hai in this order:

1. CPU if `FORCE_CPU = True`
2. CUDA if available
3. MPS on Apple Silicon
4. otherwise CPU

Internship explanation:
Yeh isliye kiya gaya so notebook different systems par chal sake, but Mac ke case mein stable path CPU nikla.

---

### 4.3 Why `TARGET_TOTAL_IMAGES` Use Kiya?

Yeh bahut important design choice hai.

Dataset mein:

- `NORMAL` images kam hain
- `PNEUMONIA` images zyada hain

Agar hum sab images direct use karte:

- model pneumonia-heavy data dekhta
- bias aa sakta tha
- evaluation misleading ho sakti thi

Isliye notebook:

- full dataset paths gather karta hai
- class-wise counts dekhta hai
- balanced subset create karta hai

Example:

If `TARGET_TOTAL_IMAGES = 1000`, then notebook tries:

- `500 NORMAL`
- `500 PNEUMONIA`

If `TARGET_TOTAL_IMAGES = 3000`, then ideally:

- `1500 NORMAL`
- `1500 PNEUMONIA`

But agar smaller class mein utne images hi nahi hain, to code automatically smaller class count ke according cap kar deta hai.

Short version:
`TARGET_TOTAL_IMAGES` runtime control + class balance dono ke liye hai. Smart move hai, random jugaad nahi.

---

### 4.4 Data Loading

Notebook:

- image file paths gather karta hai
- labels assign karta hai
- balanced dataframe create karta hai
- stratified split karta hai into:

  - `70% train`
  - `15% val`
  - `15% test`

Stratified split ka fayda:
Har split mein class distribution balanced rehta hai.

---

### 4.5 Image Transforms

Training transforms include:

- grayscale to 3 channels
- resize to `224 x 224`
- horizontal flip
- slight rotation
- brightness/contrast augmentation
- normalization with ImageNet mean/std

Evaluation transforms simpler hote hain:

- grayscale to 3 channels
- resize
- tensor conversion
- normalization

Why grayscale ko 3 channels banaya?
Because pretrained ViT RGB-style input expect karta hai. X-ray grayscale hai, so same image 3 channels mein map ki gayi.

---

### 4.6 Custom Dataset Class

Notebook ka `ChestXrayDataset`:

- dataframe se image path uthata hai
- image open karta hai
- label return karta hai
- transform apply karta hai

Output hota hai:

- image tensor
- label
- image path

Path bhi return karna useful hai, especially debugging aur sample prediction display ke liye.

---

### 4.7 DataLoader Design

Humne Mac/Jupyter issues ke baad final stable setup rakha:

- `num_workers = 0` for CPU/MPS
- CUDA par workers use ho sakte hain

Why?
Mac notebook runs mein multiprocessing error aa raha tha:

`Can't get attribute 'ChestXrayDataset' on <module '__main__'>`

Ye Jupyter + multiprocessing + custom dataset pickling issue tha.

Solution:
Single-process loading.

Thoda slower, but stable. Aur internship demo ke liye stability > fancy speed.

---

## 5. Model Kaise Banaya Gaya

### 5.1 Base Model

Humne pretrained model use kiya:

- `google/vit-base-patch16-224`

Ye originally ImageNet ke `1000` classes ke liye trained hota hai.

Humara task binary classification hai:

- `NORMAL`
- `PNEUMONIA`

So final classifier layer change hui from:

- `1000 outputs`

to:

- `2 outputs`

Isi wajah se load report mein classifier mismatch dikhta tha.

---

### 5.2 Woh `MISMATCH` Warning Kya Thi?

Message something like:

```text
classifier.weight | MISMATCH
classifier.bias   | MISMATCH
```

Iska matlab:

- pretrained backbone load ho gaya
- final classification layer size alag thi
- model ne us last layer ko reinitialize kiya

Yeh expected hai.
Yeh bug nahi tha.
Actually yahi sahi behavior tha.

In short:
Backbone ne kaam ki knowledge retain ki, classifier layer ne hamare 2-class task ke hisaab se naya start liya.

---

## 6. Training Strategy

Training ek hi shot mein nahi ki gayi. Staged fine-tuning ki gayi.

### Stage 1: Frozen Backbone

Only classifier head train hota hai.

Why?
Model ko pehle new task samajhne do without disturbing full pretrained network.

### Stage 2: Fine-tuning Last Blocks

Last few transformer blocks unfreeze kiye gaye.

Why?
Ab model task-specific features aur better learn kar sakta hai.

### Stage 3: Full Fine-tuning

Entire model unfreeze karke low learning rate par train kiya gaya.

Why?
Final refinement. Thoda polish mode. Interview style bolna ho to:
Global adaptation with careful low-LR end-to-end fine-tuning.

---

## 7. Loss, Optimizer, Metrics

Notebook uses:

- `CrossEntropyLoss`
- `AdamW`
- accuracy
- precision
- recall
- F1-score
- confusion matrix

Why F1 important?
Medical classification mein sirf accuracy dekhna enough nahi hota, especially jab false negatives dangerous ho sakte hain.

---

## 8. Early Stopping

Notebook early stopping use karta hai validation loss par.

Why?

- overfitting rokne ke liye
- unnecessary epochs avoid karne ke liye
- best model state preserve karne ke liye

Matlab model ko bol rahe:
"Bhai agar tum improve nahi kar rahe, to bas karo, energy bachao."

---

## 9. Evaluation

Training ke baad notebook:

- test loss
- test accuracy
- precision
- recall
- F1
- classification report
- confusion matrix

show karta hai.

Your completed run gave roughly:

- Accuracy: `92.67%`
- F1: `0.9252`

Balanced test set par yeh strong result hai.

---

## 10. Model Saving

Notebook checkpoint save karta hai in:

- [vit_chest_xray_classifier.pt](/Users/mayank/Downloads/Pratty-proj/vision_transformer/outputs/vit_chest_xray_classifier.pt)

Checkpoint mein hota hai:

- model weights
- config
- class names
- history

Isliye baad mein prediction script easily same model rebuild kar paati hai.

---

## 11. Attention Visualization

Notebook ke end mein attention rollout ka section bhi hai.

Problem kya aaya?
New Transformers versions mein `output_attentions` with `sdpa` attention mode issue de raha tha.

Fix:
Attention visualization ke liye separate eager-attention model use kiya gaya.

Important:
Yeh training blocker nahi tha.
Sirf last visualization step ka issue tha.

---

## 12. Mac Support Notes

Yeh section especially Mac users ke liye hai.

### 12.1 MPS Support

Apple Silicon systems par `mps` theoretically use ho sakta hai.

Humne try kiya:

- notebook ko `mps` support diya
- fallback bhi add kiya

But practical issue:

- higher workload
- larger subset
- Jupyter kernel instability

So final stable recommendation on your system:

- local training on CPU
- moderate subset size

### 12.2 Why MPS Was Crashing

Likely reasons:

- large ViT model
- notebook environment
- longer training duration
- memory pressure at higher sample count
- PyTorch/Transformers/MPS stack instability

Conclusion:
`MPS` theoretically supported tha, but your machine par stable production path CPU nikla.

### 12.3 Why CPU Was Safe

CPU path:

- slower tha
- but stable tha
- no kernel death
- completed end-to-end

Internship mein safe reproducible result usually best hota hai.

---

## 13. Prediction Script Kaise Work Karta Hai

File:
[predict_single_xray.py](/Users/mayank/Downloads/Pratty-proj/vision_transformer/predict_single_xray.py)

Iska kaam:
Ek saved model checkpoint load karo, ek X-ray image do, aur prediction le lo.

### Step-by-step

#### 1. Command-line arguments read karta hai

You pass:

- image path
- optional checkpoint path
- optional `--cpu`

#### 2. Device choose karta hai

Priority:

- forced CPU if `--cpu`
- CUDA if available
- MPS if available
- else CPU

#### 3. Checkpoint load karta hai

Checkpoint se aata hai:

- config
- weights
- class names

Then same ViT model reconstruct hota hai.

#### 4. Image preprocess hoti hai

Exactly evaluation jaisa:

- grayscale to 3 channels
- resize to model size
- tensor
- normalization

#### 5. Model prediction karta hai

- logits milte hain
- softmax apply hota hai
- highest probability class choose hoti hai

#### 6. Output print hota hai

Example:

```text
Predicted class: NORMAL
Confidence: 0.9940
NORMAL: 0.9940
PNEUMONIA: 0.0060
```

Simple, clean, deployable-style inference.

---

## 14. Non-blocking Warnings and What They Meant

### 14.1 HF Hub Warning

Message:
unauthenticated requests to HF Hub

Meaning:
Model without login/token download ho raha tha.

Impact:
Usually none for local project.
Bas download thoda slower ho sakta hai.

### 14.2 `num_labels=2` vs `id2label` map warning

Meaning:
Pretrained model `1000` labels ke liye bana tha, humne usko `2` labels ke liye adapt kiya.

Impact:
Expected, not fatal.

### 14.3 `GradScaler` FutureWarning

Meaning:
Old AMP API deprecate ho rahi thi.

Impact:
Training par koi blocker nahi.

Fix:
Notebook updated to newer `torch.amp` API.

### 14.4 `LOAD REPORT` mismatch

Meaning:
Final classifier layer resize hui.

Impact:
Expected.

### 14.5 DataLoader worker crash

Meaning:
Jupyter multiprocessing issue on Mac.

Impact:
Notebook stopped.

Fix:
`NUM_WORKERS = 0`

### 14.6 MPS kernel death

Meaning:
Hard backend instability, normal Python exception bhi nahi.

Impact:
Kernel restart.

Fix:
Prefer CPU for local stable training.

---

## 15. Why This Project Design Makes Sense

Agar internship review mein explain karna ho, you can say:

1. We used transfer learning because medical datasets are limited and ViT already has strong visual representations.
2. We balanced the dataset to reduce class bias.
3. We used staged fine-tuning to adapt the pretrained model gradually.
4. We tracked clinically relevant metrics like precision, recall, and F1, not just accuracy.
5. We added model saving and standalone inference for practical usability.
6. We adapted the notebook for local Mac stability, which improved reproducibility.

Bas. Clean and solid explanation.

---

## 16. Suggested Presentation Flow

Agar tumhe present karna ho, is order mein bolo:

1. Problem statement
2. Dataset imbalance issue
3. Why ViT
4. Data preprocessing and balancing
5. Staged fine-tuning approach
6. Evaluation metrics
7. Final performance
8. Practical deployment via single-image prediction script
9. Mac-specific engineering adjustments

Yeh flow smart lagta hai and very defendable hai.

---

## 17. Q&A

### Q1. Why did you use a Vision Transformer instead of a CNN?

ViT ne large-scale pretraining se strong visual features learn kiye hote hain. Transfer learning ke through usko chest X-ray classification task par adapt karna effective tha.

### Q2. Why convert grayscale X-rays to 3 channels?

Pretrained ViT RGB-style input expect karta hai. Grayscale image ko 3 channels mein replicate karke model input format preserve kiya gaya.

### Q3. Why not train on the full dataset directly?

Dataset imbalanced tha. Direct training se class bias aa sakta tha. Balanced subset use karne se training fair aur controlled hui.

### Q4. Why use `TARGET_TOTAL_IMAGES`?

Runtime control aur class balance dono ke liye. It makes experimentation easier and more reproducible.

### Q5. Why was `PNEUMONIA` not used fully if more images were available?

Because balanced training objective tha. Humne extra pneumonia images intentionally skip kiye to avoid bias.

### Q6. Why did you use stratified splitting?

Har split mein class proportions stable rakhne ke liye. This gives more reliable validation and testing.

### Q7. What does the classifier mismatch warning mean?

Pretrained model ka last layer `1000` classes ke liye tha. Hamara task `2` classes ka tha. So last layer reinitialized hui. Expected behavior.

### Q8. Why freeze the backbone first?

Pehle classifier ko task-specific mapping learn karne diya. Isse pretrained features immediately disturb nahi hote.

### Q9. Why unfreeze only the last few transformer blocks in stage 2?

Gradual adaptation ke liye. Pura model ekdum se unfreeze karna unstable ho sakta hai.

### Q10. Why full fine-tuning at the end?

Final task-specific refinement ke liye. Low learning rate ke saath full network align ho jata hai.

### Q11. Why is F1-score important here?

Medical classification mein class-specific mistakes important hote hain. F1 precision aur recall dono ko combine karta hai.

### Q12. Why not rely only on accuracy?

Accuracy kabhi-kabhi imbalance ya error type ko hide kar sakti hai. Precision/recall/F1 better insight dete hain.

### Q13. Why use early stopping?

Overfitting reduce karne ke liye and best validation model preserve karne ke liye.

### Q14. Why did the DataLoader fail on Mac?

Jupyter notebook environment mein multiprocessing custom dataset class ko pickle nahi kar paa raha tha.

### Q15. Why was `num_workers=0` chosen?

Because stable execution more important tha than slightly faster loading.

### Q16. Why did MPS crash?

Likely backend instability under larger workload. This was environment-specific, not a logic bug in the training pipeline.

### Q17. Is the project wrong because MPS crashed?

Bilkul nahi. Model pipeline correct thi. Platform/backend stability issue alag cheez hoti hai.

### Q18. Why did CPU succeed?

CPU slower hai but more stable tha for this exact environment and notebook setup.

### Q19. What does the prediction script add to the project?

It turns the trained model into a reusable tool. Notebook ke bahar bhi inference ho sakta hai.

### Q20. Can this project be extended further?

Yes:

- bigger balanced runs
- better augmentation
- hyperparameter tuning
- ROC-AUC and sensitivity-specificity reporting
- folder-level batch inference
- web app / Streamlit deployment

---

## 18. Final Summary

Project ne:

- pretrained ViT use kiya
- imbalanced dataset ko balanced subset mein convert kiya
- staged fine-tuning ki
- good performance achieve ki
- checkpoint save kiya
- single-image prediction script add ki
- Mac-specific runtime issues solve kiye

Shortest summary:
Research vibe bhi aa gaya, engineering vibe bhi aa gaya.

Aur haan, model ne kaam bhi kiya. Sirf slides hi pretty nahi bani.

