import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import os

# 1. Dataset size selection
desired_size = 300000

# 2. Load and sample data
df_full = pd.read_csv('/kaggle/input/330k-arabic-sentiment-reviews/arabic_sentiment_reviews.csv').dropna(subset=['content', 'label'])
if desired_size < len(df_full):
    df_full = df_full.sample(desired_size, random_state=42).reset_index(drop=True)
else:
    df_full = df_full.reset_index(drop=True)
texts = df_full['content'].astype(str).tolist()
labels = df_full['label'].astype(int).values

# 3. Sequence preprocessing
sent_words = [sent.split() for sent in texts]
max_sentence_len = min(max(len(words) for words in sent_words), 30)
max_word_len = min(max(max(len(word) for word in sent) for sent in sent_words), 8)
all_chars = set(char for sent in sent_words for word in sent for char in word)
char2idx = {c: i+1 for i, c in enumerate(sorted(all_chars))}

def encode_text(sent, max_sentence_len, max_word_len, char2idx):
    encoded = np.zeros((max_sentence_len, max_word_len), dtype='int32')
    for i, word in enumerate(sent[:max_sentence_len]):
        for j, char in enumerate(word[:max_word_len]):
            encoded[i, j] = char2idx.get(char, 0)
    return encoded

# 4. Split to train/validation
train_df, valid_df = train_test_split(df_full, test_size=0.2, stratify=df_full['label'], random_state=42)

class TextSequence(Sequence):
    def __init__(self, df, batch_size, max_sentence_len, max_word_len, char2idx):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_sentence_len = max_sentence_len
        self.max_word_len = max_word_len
        self.char2idx = char2idx
        self.indices = np.arange(len(df))
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        batch_texts = batch_df['content'].astype(str).tolist()
        batch_labels = batch_df['label'].astype(int).values
        batch_inputs = np.array([encode_text(sent.split(), self.max_sentence_len, self.max_word_len, self.char2idx) for sent in batch_texts])
        return batch_inputs, batch_labels

batch_size = 64
train_gen = TextSequence(train_df, batch_size, max_sentence_len, max_word_len, char2idx)
valid_gen = TextSequence(valid_df, batch_size, max_sentence_len, max_word_len, char2idx)

# 5. SDE Layer definition
class SDELayer(layers.Layer):
    def __init__(self, hidden_dim, dt=1.0, noise_std=0.2, steps=5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.noise_std = noise_std
        self.steps = steps
        self.dense_f = layers.Dense(hidden_dim, activation='tanh')
        self.dense_g = layers.Dense(hidden_dim, activation='tanh')
    def call(self, h, training=None):
        for _ in range(self.steps):
            drift = self.dense_f(h) * self.dt
            diffusion = self.dense_g(h) * tf.random.normal(tf.shape(h), mean=0.0, stddev=self.noise_std)
            h = h + drift + diffusion
        return h

# 6. Model construction
input_layer = layers.Input(shape=(max_sentence_len, max_word_len))
emb = layers.TimeDistributed(layers.Embedding(len(char2idx)+1, 32, mask_zero=True))(input_layer)
cnn1 = layers.TimeDistributed(layers.Conv1D(128, 7, activation='relu', padding='same'))(emb)
cnn2 = layers.TimeDistributed(layers.Conv1D(64, 4, activation='relu', padding='same'))(cnn1)
cnn_pooled = layers.TimeDistributed(layers.GlobalMaxPooling1D())(cnn2)
sde_out = SDELayer(64, name='custom_sde_layer')(cnn_pooled)
flat = layers.Flatten()(sde_out)
dense = layers.Dense(32, activation='relu')(flat)
output = layers.Dense(1, activation='sigmoid')(dense)
model = models.Model(inputs=input_layer, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 7. Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(train_gen, epochs=20, validation_data=valid_gen, callbacks=[early_stop], verbose=1)

# Plot accuracy/loss

import matplotlib.pyplot as plt

# Accuracy curve
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy', fontsize=18)              # Larger title
plt.ylabel('Accuracy', fontsize=16)                   # Larger y-axis label
plt.xlabel('Epoch', fontsize=16)                      # Larger x-axis label
plt.xticks(fontsize=14)                               # Larger x-axis numbers
plt.yticks(fontsize=14)                               # Larger y-axis numbers
plt.legend(loc='lower right', fontsize=14)            # Larger legend
plt.savefig('/kaggle/working/accuracy_curve_sde.png')
plt.show()

# Loss curve
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss', fontsize=18)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.savefig('/kaggle/working/loss_curve_sde.png')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Example: Larger font sizes for all plots
plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

# Confusion matrix
cm = confusion_matrix(valid_y, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", annot_kws={"size": 16})
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.title('Confusion Matrix SDE', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('/kaggle/working/confusion_matrix_sde.png', bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(valid_y, preds_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(valid_y, preds_prob):.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve SDE', fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('/kaggle/working/roc_curve_sde.png', bbox_inches='tight')
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(valid_y, preds_prob)
plt.figure(figsize=(6,5))
plt.plot(recall_vals, precision_vals, label='PR curve', linewidth=2)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision-Recall Curve SDE', fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('/kaggle/working/pr_curve_sde.png', bbox_inches='tight')
plt.show()


# 9. Latent Trajectory Visualization (PCA)
N_samples = 40
K_steps = 5
latent_dim = 64

def extract_sde_trajectory(sde_layer, encoded_sample):
    h = tf.convert_to_tensor(encoded_sample, dtype=tf.float32)
    h = tf.reshape(h, (1, latent_dim))
    trajectory = []
    for _ in range(K_steps):
        drift = sde_layer.dense_f(h) * sde_layer.dt
        diffusion = sde_layer.dense_g(h) * tf.random.normal(tf.shape(h), mean=0.0, stddev=sde_layer.noise_std)
        h = h + drift + diffusion
        trajectory.append(h.numpy().reshape(-1, latent_dim)[0])
    return np.stack(trajectory)

indices = np.random.choice(len(valid_df), N_samples, replace=False)
sample_encoded = []
sde_input_model = models.Model(inputs=model.input, outputs=model.get_layer('custom_sde_layer').input)
for i in indices:
    x = encode_text(valid_df.iloc[i]['content'].split(), max_sentence_len, max_word_len, char2idx)
    x = np.expand_dims(x, axis=0)
    sde_in = sde_input_model.predict(x)
    pooled = sde_in.squeeze()
    if pooled.ndim == 2 and pooled.shape[1] == latent_dim:
        pooled_vec = np.mean(pooled, axis=0)
    elif pooled.ndim == 1 and pooled.shape[0] == latent_dim:
        pooled_vec = pooled
    else:
        print('BAD SHAPE:', pooled.shape)
        pooled_vec = np.zeros(latent_dim)
    sample_encoded.append(pooled_vec)

sample_labels = valid_df.iloc[indices]['label'].astype(int).tolist()
sde_layer = model.get_layer('custom_sde_layer')
traj = np.array([extract_sde_trajectory(sde_layer, sample) for sample in sample_encoded])
X_flat = traj.reshape(-1, latent_dim)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_flat)
plt.figure(figsize=(16,16))
for i in range(N_samples):
    pts = X_2d[i*K_steps:(i+1)*K_steps]
    color = 'red' if sample_labels[i] == 1 else 'blue'
    plt.plot(pts[:,0], pts[:,1], marker='o', color=color, alpha=0.7)
plt.title("Latent Trajectory Visualization via PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig('/kaggle/working/latent_trajectory_pca.png')
plt.show()

model.save('/kaggle/working/sentiment_SDE_model.h5')
print(os.listdir('/kaggle/working'))



