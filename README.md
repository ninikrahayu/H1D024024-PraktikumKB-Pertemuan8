## Klasifikasi Gambar Rock, Paper, Scissors Menggunakan CNN
Program ini merupakan implementasi **Convolutional Neural Network (CNN)** menggunakan **TensorFlow** dan **Keras** untuk mengklasifikasikan gambar ke dalam tiga kelas, yaitu:
- Rock
- Paper
- Scissors

Dataset yang digunakan disimpan dalam folder `rockpaperscissors`. Dataset yang digunakan pada program ini adalah dataset **Rock Paper Scissors Images** yang diperoleh dari Kaggle. Dataset dapat diunduh melalui link berikut:
https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/download

---

## Library yang Digunakan
### 1. NumPy
Digunakan untuk pengolahan data numerik dalam bentuk array.
### 2. Pandas
Digunakan sebagai library pendukung untuk pengolahan data.
### 3. TensorFlow
Digunakan sebagai library utama untuk membangun, melatih, dan menjalankan model deep learning.
### 4. Keras
Digunakan untuk membuat arsitektur model CNN dengan lebih sederhana melalui `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, dan `Dense`.
### 5. ImageDataGenerator
Digunakan untuk membaca gambar dari folder, melakukan preprocessing gambar, membagi data training dan validation, serta memberi label otomatis berdasarkan nama folder.

---

## Struktur Dataset
Dataset harus diletakkan dalam folder:
```text
rockpaperscissors
````
Struktur foldernya:
```text
rockpaperscissors/
├── paper/
├── rock/
└── scissors/
```
Setiap folder berisi gambar sesuai kelasnya. Folder `paper` berisi gambar paper, folder `rock` berisi gambar rock, dan folder `scissors` berisi gambar scissors.

---

## Penjelasan Kode
### 1. Import Library
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
Bagian ini digunakan untuk memanggil library yang dibutuhkan. TensorFlow dan Keras digunakan untuk membuat model CNN, sedangkan ImageDataGenerator digunakan untuk membaca dan memproses gambar dari folder dataset.

---

### 2. Menentukan Path Dataset
```python
dataset_path = "./rockpaperscissors"
```
Kode ini digunakan untuk menentukan lokasi dataset. Tanda `./` berarti folder `rockpaperscissors` berada satu lokasi dengan file program Python.

---

### 3. Preprocessing Data dengan ImageDataGenerator
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```
Bagian ini digunakan untuk mempersiapkan data gambar sebelum masuk ke model.
Penjelasan parameter:
* `rescale=1./255` digunakan untuk mengubah nilai piksel gambar dari rentang 0-255 menjadi 0-1.
* `validation_split=0.2` digunakan untuk membagi dataset menjadi 80% data training dan 20% data validation.

---

### 4. Membuat Data Training
```python
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)
```
Kode ini digunakan untuk mengambil data training dari folder dataset.
Penjelasan parameter:
* `dataset_path` adalah lokasi dataset.
* `target_size=(150, 150)` digunakan untuk mengubah ukuran semua gambar menjadi 150 x 150 piksel.
* `batch_size=32` berarti gambar diproses sebanyak 32 gambar dalam satu batch.
* `class_mode='categorical'` digunakan karena klasifikasi memiliki lebih dari dua kelas.
* `subset='training'` berarti data yang digunakan adalah bagian training.

---

### 5. Membuat Data Validation
```python
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)
```
Kode ini digunakan untuk mengambil data validation dari folder dataset. Data validation digunakan untuk mengevaluasi performa model selama proses training.

---

### 6. Membuat Model CNN
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])
```
Model dibuat menggunakan `Sequential`, yaitu model yang layer-nya disusun secara berurutan.
Struktur model:
#### Conv2D
Layer `Conv2D` digunakan untuk mengekstraksi fitur dari gambar, seperti bentuk, garis, tepi, dan pola tertentu.
```python
Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3))
```
* `32` adalah jumlah filter.
* `(3,3)` adalah ukuran kernel.
* `activation='relu'` digunakan untuk membantu model mempelajari pola non-linear.
* `input_shape=(150,150,3)` berarti gambar input berukuran 150 x 150 piksel dengan 3 channel warna RGB.

#### MaxPooling2D
Layer `MaxPooling2D` digunakan untuk mengecilkan ukuran feature map agar proses training lebih ringan.
```python
MaxPooling2D(2,2)
```

#### Flatten
Layer `Flatten` digunakan untuk mengubah data hasil convolution dan pooling menjadi bentuk satu dimensi agar bisa masuk ke Dense layer.

#### Dense
Layer `Dense(512, activation='relu')` digunakan sebagai fully connected layer untuk memproses fitur yang sudah diekstraksi.

#### Output Layer
```python
Dense(3, activation='softmax')
```
Output layer memiliki 3 neuron karena terdapat 3 kelas, yaitu rock, paper, dan scissors. Aktivasi `softmax` digunakan untuk menghasilkan probabilitas dari masing-masing kelas.

---

### 7. Menampilkan Ringkasan Model
```python
model.summary()
```
Kode ini digunakan untuk menampilkan struktur model, bentuk output setiap layer, dan jumlah parameter yang dilatih.

---

### 8. Compile Model
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```
Bagian ini digunakan untuk mengatur model sebelum dilatih.
Penjelasan:
* `loss='categorical_crossentropy'` digunakan karena kasus ini merupakan klasifikasi multi-kelas.
* `optimizer='adam'` digunakan untuk memperbarui bobot model agar hasil prediksi semakin baik.
* `metrics=['accuracy']` digunakan untuk menampilkan nilai akurasi selama training.

---

### 9. Training Model
```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
```
Kode ini digunakan untuk melatih model menggunakan data training.
Penjelasan:
* `train_generator` adalah data training.
* `validation_data=validation_generator` digunakan untuk mengecek performa model pada data validation.
* `epochs=10` berarti model dilatih sebanyak 10 kali putaran.

---

### 10. Evaluasi Model
```python
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')
```
Kode ini digunakan untuk mengevaluasi model menggunakan data validation.
Hasil evaluasi berupa:
* `Validation loss`, yaitu nilai kesalahan model.
* `Validation accuracy`, yaitu tingkat ketepatan model dalam mengklasifikasikan gambar.

---

### 11. Prediksi
```python
predictions = model.predict(validation_generator)
print(predictions)
```
Kode ini digunakan untuk melakukan prediksi terhadap data validation.
Output yang muncul berupa nilai probabilitas untuk setiap kelas. Karena terdapat 3 kelas, maka setiap gambar akan menghasilkan 3 nilai probabilitas.
Contoh:
```text
[0.98, 0.01, 0.01]
```
Artinya model memprediksi gambar tersebut paling besar kemungkinan masuk ke kelas pertama.

---

