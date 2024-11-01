# Laporan Proyek Machine Learning - Dzulfikri Adjmal

## Domain Proyek

Era digital telah membawa perubahan besar pada berbagai sektor, termasuk sektor transportasi. Salah satu sektor transportasi yang mengalami perubahan besar adalah maskapai penerbangan. Jumlah data yang dihasilkan oleh maskapai penerbangan sangat besar dan bervariasi. Data tersebut dapat digunakan untuk memahami kepuasan penumpang, memprediksi keterlambatan penerbangan, dan lain sebagainya.

Kepuasan pelanggan merupakan bagian yang berhubungan dengan penciptaan nilai pelanggan yang memberikan manfaat bagi perusahaan, diantaranya hubungan perusahaan dengan pelanggan menjadi harmonis dan terciptanya suatu rekomendasi dari mulut ke mulut yang menguntungkan perusahaan[3]. Sedangkan menurut Kamus Besar Bahasa Indonesia, puas adalah lebih dari cukup[1]. Dengan kata lain, kepuasan pelanggan adalah suatu kondisi dimana pelanggan merasa lebih dari cukup terhadap produk atau layanan yang diberikan oleh perusahaan.

Masalah kepuasan pelanggan dalam sektor penerbangan menjadi sangat penting untuk diselesaikan karena kepuasan pelanggan berkaitan langsung dengan keberlanjutan bisnis dan reputasi perusahaaan. Kepuasan pelanggan menjadi salah satu faktor pertimbangan utama dalam memilih maskapai dan menjadi pembeda yang membuat suatu maskapai lebih unggul dibandingkan maskapai lainnya. Oleh karena itu, perusahaan maskapai penerbangan perlu memahami kepuasan pelanggan dengan baik agar dapat meningkatkan kualitas layanan dan mempertahankan pelanggan.

Dengan perkembangan teknologi yang semakin cepat, perusahaan dapat menggunakan teknologi machine learning untuk mengklasifikasikan kepuasan pelanggan berdasarkan data yang dimiliki. Hasil dari pemrosesan data dan klasifikasi kepuasan pelanggan dapat membantu perusahaan dalam mengambil keputusan yang lebih baik dan tepat dalam meningkatkan kualitas layanan dan kepuasan pelanggan.

Proyek ini bertujuan untuk mengembangkan sebuah model machine learning yang dapat mengklasifikasikan kepuasan dan ketidakpuasan penumpang maskapai. Data-data yang digunakan meliputi beberapa layanan yang disediakan oleh maskapai, seperti pelayanan kabin, pelayanan makanan, penanganan bagasi, dan lain sebagainya. Dengan model machine learning yang dikembangkan, perusahaan maskapai penerbangan dapat memahami kepuasan pelanggan dan meningkatkan kualitas layanan yang diberikan.

## Business Understanding

### Problem Statements

Dari latar belakang di atas, proyek ini akan menjawab beberapa pertanyaan berikut:

- Model machine learning apa yang paling baik dari algoritma K-Nearest Neighbors, Random Forest, dan CatBoost dalam mengklasifikasikan kepuasan penumpang?
- Apakah layanan Inflight Wifi Service, Seat Comfort, dan Inflight Entertainment berkontribusi terhadap klasifikasi kepuasan penumpang?
- Bagaimana langkah yang dapat diambil oleh maskapai untuk meningkatkan kualitas layanan maskapai penerbangan berdasarkan hasil klasifikasi kepuasan penumpang?

### Goals

Proyek ini memiliki tujuan sebagai berikut:

- Mengembangkan model machine learning yang dapat mengklasifikasikan kepuasan penumpang.
- Mengetahui kontribusi layanan Inflight Wifi Service, Seat Comfort, dan Inflight Entertainment terhadap klasifikasi kepuasan penumpang.
- Bagaimana implementasi machine learning dalam mengklasifikasikan kepuasan penumpang dapat membantu perusahaan maskapai penerbangan dalam meningkatkan kualitas layanan.

### Solution statements

- Menggunakan algoritma K-Nearest Neighbors, Random Forest, dan CatBoost untuk mengklasifikasikan kepuasan penumpang.
- Menerapkan pra-pemrosesan data, pembersihan data, dan penyeimbangan kelas untuk meningkatkan performa model.

## Data Understanding

Data yang digunakan merupakan data yang diambil dari [Kaggle: Airline Passenger Satisfaction](https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction) yang berisi mengenai kepuasan penumpang maskapai penerbangan. Dataset tersebut sudah dibagi menjadi data latih dan data uji dengan proporsi 80:20. Data ini terdiri dari beberapa fitur yang dapat digunakan untuk mengklasifikasikan kepuasan penumpang.

### Variabel-variabel pada dataset adalah sebagai berikut

- Gender: Jenis kelamin penumpang (Female, Male)
- Customer Type: Tipe pelanggan (Loyal customer, disloyal customer)
- Age: Usia penumpang
- Type of Travel: Tipe perjalanan (Personal Travel, Business Travel)
- Class: Kelas penerbangan (Business, Eco, Eco Plus)
- Flight Distance: Jarak penerbangan
- Inflight wifi service: Layanan wifi selama penerbangan (0: Not Applicable; 1-5)
- Departure/Arrival time convenient: Kepuasan terhadap waktu keberangkatan/kedatangan (0: Not Applicable; 1-5)
- Ease of Online booking: Kemudahan dalam pemesanan tiket online (0: Not Applicable; 1-5)
- Gate location: Lokasi gerbang keberangkatan (0: Not Applicable; 1-5)
- Food and drink: Kepuasan terhadap makanan dan minuman (0: Not Applicable; 1-5)
- Online boarding: Kepuasan terhadap proses boarding online (0: Not Applicable; 1-5)
- Seat comfort: Kepuasan terhadap kenyamanan kursi (0: Not Applicable; 1-5)
- Inflight entertainment: Kepuasan terhadap hiburan selama penerbangan (0: Not Applicable; 1-5)
- On-board service: Kepuasan terhadap layanan di dalam pesawat (0: Not Applicable; 1-5)
- Leg room service: Kepuasan terhadap ruang kaki (0: Not Applicable; 1-5)
- Baggage handling: Kepuasan terhadap penanganan bagasi (0: Not Applicable; 1-5)
- Check-in service: Kepuasan terhadap layanan check-in (0: Not Applicable; 1-5)
- Inflight service: Kepuasan terhadap layanan selama penerbangan (0: Not Applicable; 1-5)
- Cleanliness: Kepuasan terhadap kebersihan pesawat (0: Not Applicable; 1-5)
- Departure Delay in Minutes: Keterlambatan keberangkatan dalam menit
- Arrival Delay in Minutes: Keterlambatan kedatangan dalam menit
- Satisfaction: Kepuasan penumpang (Satisfied, Neutral or dissatisfied)

### Exploratory Data Analysis

Untuk lebih mengenali dataset yang digunakan dan mengetahui karakteristik dari data, dilakukan eksplorasi data dengan beberapa metode berikut:

- Melihat ukuran data
- Melihat statistik deskriptif dari data
- Melihat distribusi data
- Melihat korelasi antar variabel
- Mengecek data missing value dan membersihkan data missing value
- Mengecek duplikasi dan menghapus data duplikasi
- Mengecek data outlier dan menangani data outlier

#### Ukuran Data

Jumlah baris dan kolom

| Data | Jumlah Baris | Jumlah Kolom | Missing Value | Duplicate Value |
|------|--------------|--------------|---------------| --------------- |
| Train| 103904| 24| 310|0|
| Test | 25976| 24| 83|0|

#### Statistik Deskriptif

Statistik deskriptif dari data latih

|       | Age        | Flight Distance | Inflight wifi service | Departure/Arrival time convenient | Ease of Online booking | Gate location | Food and drink | Online boarding | Seat comfort | Inflight entertainment | On-board service | Leg room service | Baggage handling | Checkin service | Inflight service | Cleanliness | Departure Delay in Minutes | Arrival Delay in Minutes |
|-------|------------|-----------------|------------------------|-----------------------------------|------------------------|---------------|----------------|-----------------|--------------|------------------------|------------------|------------------|------------------|-----------------|------------------|-------------|----------------------------|--------------------------|
| count | 103904.00  | 103904.00       | 103904.00             | 103904.00                         | 103904.00             | 103904.00     | 103904.00      | 103904.00       | 103904.00    | 103904.00             | 103904.00        | 103904.00        | 103904.00        | 103904.00       | 103904.00        | 103904.00   | 103904.00                   | 103594.00                |
| mean  | 39.38      | 1189.45         | 2.73                  | 3.06                              | 2.76                  | 2.98          | 3.20           | 3.25            | 3.44         | 3.36                  | 3.38             | 3.35             | 3.63             | 3.30            | 3.64             | 3.29        | 14.82                       | 15.18                    |
| std   | 15.11      | 997.15          | 1.33                  | 1.53                              | 1.40                  | 1.28          | 1.33           | 1.35            | 1.32         | 1.33                  | 1.29             | 1.32             | 1.18             | 1.27            | 1.18             | 1.31        | 38.23                       | 38.70                    |
| min   | 7.00       | 31.00           | 0.00                  | 0.00                              | 0.00                  | 0.00          | 0.00           | 0.00            | 0.00         | 0.00                  | 0.00             | 0.00             | 1.00             | 0.00            | 0.00             | 0.00        | 0.00                        | 0.00                     |
| 25%   | 27.00      | 414.00          | 2.00                  | 2.00                              | 2.00                  | 2.00          | 2.00           | 2.00            | 2.00         | 2.00                  | 2.00             | 2.00             | 3.00             | 3.00            | 3.00             | 2.00        | 0.00                        | 0.00                     |
| 50%   | 40.00      | 843.00          | 3.00                  | 3.00                              | 3.00                  | 3.00          | 3.00           | 3.00            | 4.00         | 4.00                  | 4.00             | 4.00             | 4.00             | 3.00            | 4.00             | 3.00        | 0.00                        | 0.00                     |
| 75%   | 51.00      | 1743.00         | 4.00                  | 4.00                              | 4.00                  | 4.00          | 4.00           | 4.00            | 5.00         | 4.00                  | 4.00             | 4.00             | 5.00             | 4.00            | 5.00             | 4.00        | 12.00                       | 13.00                    |
| max   | 85.00      | 4983.00         | 5.00                  | 5.00                              | 5.00                  | 5.00          | 5.00           | 5.00            | 5.00         | 5.00                  | 5.00             | 5.00             | 5.00             | 5.00            | 5.00             | 5.00        | 1592.00                     | 1584.00                  |

Statistik deskriptif dari data uji

|       | Age        | Flight Distance | Inflight wifi service | Departure/Arrival time convenient | Ease of Online booking | Gate location | Food and drink | Online boarding | Seat comfort | Inflight entertainment | On-board service | Leg room service | Baggage handling | Checkin service | Inflight service | Cleanliness | Departure Delay in Minutes | Arrival Delay in Minutes |
|-------|------------|-----------------|------------------------|-----------------------------------|------------------------|---------------|----------------|-----------------|--------------|------------------------|------------------|------------------|------------------|-----------------|------------------|-------------|----------------------------|--------------------------|
| count | 25976.00   | 25976.00        | 25976.00               | 25976.00                          | 25976.00               | 25976.00      | 25976.00       | 25976.00        | 25976.00     | 25976.00               | 25976.00         | 25976.00         | 25976.00         | 25976.00        | 25976.00         | 25976.00    | 25976.00                    | 25893.00                 |
| mean  | 39.62      | 1193.79         | 2.72                   | 3.05                              | 2.76                   | 2.98          | 3.22           | 3.26            | 3.45         | 3.36                   | 3.39             | 3.35             | 3.63             | 3.31            | 3.65             | 3.29        | 14.31                       | 14.74                    |
| std   | 15.14      | 998.68          | 1.34                   | 1.53                              | 1.41                   | 1.28          | 1.33           | 1.36            | 1.32         | 1.34                   | 1.28             | 1.32             | 1.18             | 1.27            | 1.18             | 1.32        | 37.42                       | 37.52                    |
| min   | 7.00       | 31.00           | 0.00                   | 0.00                              | 0.00                   | 1.00          | 0.00           | 0.00            | 1.00         | 0.00                   | 0.00             | 0.00             | 1.00             | 1.00            | 0.00             | 0.00        | 0.00                        | 0.00                     |
| 25%   | 27.00      | 414.00          | 2.00                   | 2.00                              | 2.00                   | 2.00          | 2.00           | 2.00            | 2.00         | 2.00                   | 2.00             | 2.00             | 3.00             | 3.00            | 3.00             | 2.00        | 0.00                        | 0.00                     |
| 50%   | 40.00      | 849.00          | 3.00                   | 3.00                              | 3.00                   | 3.00          | 3.00           | 4.00            | 4.00         | 4.00                   | 4.00             | 4.00             | 4.00             | 3.00            | 4.00             | 3.00        | 0.00                        | 0.00                     |
| 75%   | 51.00      | 1744.00         | 4.00                   | 4.00                              | 4.00                   | 4.00          | 4.00           | 4.00            | 5.00         | 4.00                   | 4.00             | 4.00             | 5.00             | 4.00            | 5.00             | 4.00        | 12.00                       | 13.00                    |
| max   | 85.00      | 4983.00         | 5.00                   | 5.00                              | 5.00                   | 5.00          | 5.00           | 5.00            | 5.00         | 5.00                   | 5.00             | 5.00             | 5.00             | 5.00            | 5.00             | 5.00        | 1128.00                     | 1115.00                  |

#### Visualisasi Data

Visualisasi data dapat membantu dalam memahami karakteristik data dan menemukan pola yang terdapat dalam data. Beberapa visualisasi data yang dapat dilakukan adalah sebagai berikut:

- Boxplot untuk melihat outlier pada data

Boxplot data latih
![Outlier data train](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/train_boxplot_outlier.png?raw=true)
 Boxplot data uji
![Outlier data test](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/test_boxplot_outlier.png?raw=true)

Boxplot merupakan salah satu metode yang digunakan untuk menampilkan distribusi data berdasarkan kuartil. Pembuatan boxplot didasarkan pada nilai minimum, kuartil bawah (Q1), median (Q2), kuartil atas (Q3), dan nilai maksimum dari data. Boxplot dapat menunjukkan adanya data outlier yang berada di luar batas atas dan batas bawah dari data yang dapat dilihat pada garis berbentuk T pada boxplot.

- Histogram untuk melihat distribusi data

Histogram pada data *Inflight wifi service*, *Inflight entertainment*, *On-Board Service*, dan *Seat comfort* terhadap *flight distance*.

![Histogram flight distance](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/histplot_flight_distance.png?raw=true)

Histogram merupakan salah satu metode visualisasi yang digunakan untuk menampilkan distribusi data. Histogram menunjukkan frekuensi kemunculan data pada interval tertentu. Histogram dapat membantu dalam mengetahui distribusi data dan menemukan pola yang terdapat dalam data.

- Heatmap untuk melihat korelasi antar variabel

![heatmap](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/heatmap_correlation.png?raw=true)

Heatmap merupakan metode visualisasi yang paling sering digunakan untuk menampilkan korelasi antar variabel. Di dalam heatmap, korelasi antar variabel ditampilkan dalam bentuk warna. Warna dari heatmap sendiri tergantung kepada nilai korelasi antar variabel tersebut. Pada proyek ini, korelasi yang mendekati 1 hingga 1 menunjukkan hubungan positif dan diberi warna yang semakin gelap terang, sedangkan korelasi yang mendekati -1 hingga -1 menunjukkan hubungan negatif dan diberi warna terang.

- Barplot untuk melihat rata-rata kepuasan penumpang terhadap variabel lainnya

![rata-rata rating](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/plot_avg_rating.png?raw=true)

## Data Preparation

Data preparation merupakan tahapan yang penting dalam proses pemodelan machine learning. Pada tahapan ini, data akan dipersiapkan agar dapat digunakan dalam proses pemodelan. Beberapa tahapan yang dilakukan dalam data preparation adalah sebagai berikut:

- Penanganan missing value

    Pada tahapan ini, data yang memiliki missing value akan ditangani dengan beberapa metode, seperti menghapus data missing value, mengisi data missing value dengan nilai rata-rata, median, modus, atau menggunakan metode imputasi lainnya.

- Penanganan data outlier

    Outlier adalah nilai yang berbeda dengan nilai yang lainnya pada dataset. Outlier memiliki dampak yang berpengaruh terhadap performa model Machine Learning[]. Outlier dapat ditangani dengan beberapa metode, seperti menghapus data outlier, melakukan transformasi data seperti log atau square root, atau menggunakan metode lainnya.

- Penanganan data kategorikal

    Dalam melakukan pemodelan Machine Learning, komputer hanya dapat memahami data dalam bentuk angka. Oleh karena itu, data kategorikal perlu diubah menjadi data numerik agar dapat digunakan dalam proses pemodelan. Data kategorikal dapat diubah menjadi data numerik dengan metode one-hot encoding, label encoding, atau metode lainnya.

- Penyeimbangan kelas

    Pada tahapan ini, data yang memiliki ketidakseimbangan kelas akan ditangani dengan beberapa metode, seperti oversampling, undersampling, atau menggabungkan kedua metode tersebut. Proses ini dilakukan agar model machine learning tidak cenderung memprediksi kelas mayoritas. Tentunya, penyeimbangan kelas akan mempengaruhi performa model machine learning yang dikembangkan.

### Penanganan Missing Value

Pada proyek ini, missing value pada data akan ditangani dengan menghapus data yang memiliki missing value. Hal ini dilakukan karena jumlah missing value yang kecil dan tidak signifikan terhadap data.

| Data | Jumlah Missing Value Sebelum | Jumlah Missing Value Setelah |
|------|------------------------------|------------------------------|
| Train| 310                          | 0                            |
| Test | 83                           | 0                            |

### Penanganan Outlier

Penanganan outlier pada data dilakukan dengan metode transformasi data menggunakan logaritma. Hal ini dilakukan untuk mengurangi dampak outlier terhadap performa model machine learning. Transformasi menggunakan logaritma adalah salah satu yang efektif untuk mengatasi outlier tanpa membuang data yang masih penting. Logaritma yang diterapkan adalah:

$y = log({x})$

![boxplot_train_setelah_transformasi](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/train_boxplot_clean.png?raw=true)

![boxplot_test_setelah_transformasi](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/test_boxplot_clear.png?raw=true)

### Penanganan Data Kategorikal

Penanganan data kategorikal dilakukan dengan mengubah data menjadi numerik yang disebut dengan istilah encoding. Pada proyek ini, data kategorikal diubah menjadi numerik dengan metode label encoding. Label encoding adalah metode yang mengubah data kategorikal menjadi data numerik berdasarkan urutan data.

| Variabel        | Kategori                  |
|-----------------|-----------------------------|
| Gender          | Male, Female                |
| Customer Type   | Loyal Customer, disloyal Customer |
| Type of Travel  | Personal Travel, Business travel |
| Class           | Eco Plus, Business, Eco     |
| Satisfaction    | neutral or dissatisfied, satisfied |

Setelah dilakukan encoding, data kategorikal akan diubah menjadi data numerik seperti berikut:

| Variabel       | Kategori                        |
|----------------|---------------------------------|
| Gender         | 1: Male, 0: Female              |
| Customer Type  | 0: Loyal Customer, 1: Disloyal Customer |
| Type of Travel | 1: Personal Travel, 0: Business Travel |
| Class          | 2: Eco Plus, 0: Business, 1: Eco |
| Satisfaction   | 0: Neutral or Dissatisfied, 1: Satisfied |

### Penyeimbangan Kelas

Kelas yang tidak seimbang dapat mempengaruhi performa model machine learning, model akan cenderung memprediksi kelas mayoritas. Oleh karena itu, penyeimbangan kelas perlu dilakukan agar model machine learning dapat memprediksi kelas dengan baik. Terdapat beberapa teknik untuk menyeimbangkan kelas, diantaranya adalah oversampling, undersampling, dan gabungan dari kedua metode tersebut. Pada proyek ini, penyeimbangan kelas dilakukan dengan metode oversampling menggunakan SMOTE (Synthetic Minority Over-sampling Technique).

![imbalance class](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/imbalance_class.png?raw=true)

![balance class](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/balance_class.png?raw=true)

## Modeling

Model yang digunakan pada proyek ini adalah algoritma K-Nearest Neighbors, Random Forest, dan CatBoost. Ketiga model tersebut dapat dijabarkan sebagai berikut:

| Algoritma | Keterangan | Kelebihan | Kekurangan |
|-----------|------------|-----------|------------|
| K-Nearest Neighbors | Algoritma klasifikasi berbasis instance-based learning yang menggunakan jarak untuk mengklasifikasikan data | - Sederhana dan mudah diimplementasikan <br> - Dapat menangani data yang memilih kelas yang banyak | - Sensitif terhadap outlier <br> - Memerlukan waktu komputasi yang lama untuk data yang besar |
| Random Forest | Algoritma klasifikasi berbasis ensemble learning yang menggunakan decision tree | - Tahan terhadap data noise <br> - Memiliki akurasi yang sangat tinggi karena menggunakan pendekatan gabungan Decision Tree <br> | - Komputasi yang sangat kompleks <br> - Tidak dapat menghasilkan model yang mudah diinterpretasi |
| CatBoost | Algoritma klasifikasi berbasis gradient boosting yang menggunakan decision tree | - Sangat efisien untuk mengatur data kategorikal tanpa perlu encoding <br> - Mengimplementasikan beberapa teknik untuk mencegah overfitting | - Menghasilkan model yang cukup besar |

Ketiga model tersebut dilatih dengan menggunakan parameter default yang sudah ditentukan pada library `scikit-learn` dan menggunakan data yang sudah dipersiapkan dengan proporsi 80:20.

## Evaluation

Metrik yang digunakan pada proyek ini adalah akurasi (*accuracy*) dan presisi (*precision*). Berikut adalah penjelasan mengenai metrik evaluasi yang digunakan:

- **Akurasi (*Accuracy*)**: Akurasi adalah rasio prediksi yang benar (positif dan negatif) dengan keseluruhan data. Akurasi dapat dihitung dengan rumus berikut:

    $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

    Dengan:
    - TP: True Positive
    - TN: True Negative
    - FP: False Positive
    - FN: False Negative

- **Presisi (*Precision*)**: Presisi adalah rasio prediksi positif yang benar dibandingkan dengan keseluruhan prediksi positif. Presisi dapat dihitung dengan rumus berikut:
    
    $Precision = \frac{TP}{TP + FP}$

    Dengan:
    - TP: True Positive
    - FP: False Positive

Nilai dari TP, TN, FP, dan FN dapat dihitung dari confusion matrix. Confusion matrix adalah tabel yang digunakan untuk mengukur performa model machine learning. Confusion matrix terdiri dari empat bagian, yaitu True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN). Berikut adalah confusion matrix dari ketiga model:

- Confusion Matrix K-Nearest Neighbors

    ![Confusion Matrix KNN](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/cm_knn.png?raw=true)

    Perhitungan:

    - True Positive (TP): 9922
    - True Negative (TN): 10080
    - False Positive (FP): 4180
    - False Negative (FN): 4476

    Akurasi: $\frac{9922 + 10080}{9922 + 10080 + 4180 + 4476} = 0.70$

    Presisi: $\frac{9922}{9922 + 4180} = 0.69$

- Confusion Matrix Random Forest

    ![Confusion Matrix Random Forest](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/cm_rf.png?raw=true)

    Perhitungan:

    - True Positive (TP): 13332
    - True Negative (TN): 14198
    - False Positive (FP): 4180
    - False Negative (FN): 4476

    Akurasi: $\frac{13332 + 14198}{13332 + 14198 + 4180 + 4476} = 0.96$

    Presisi: $\frac{13332}{13332 + 4180} = 0.97$

- Confusion Matrix CatBoost

    ![Confusion Matrix Catboost](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/cm_cb.png?raw=true)

    Perhitungan:

    - True Positive (TP): 13442
    - True Negative (TN): 14229
    - False Positive (FP): 770
    - False Negative (FN): 358

    Akurasi: $\frac{13442 + 14229}{13442 + 14229 + 770 + 358} = 0.97$

    Presisi: $\frac{13442}{13442 + 770} = 0.98$

Jika berdasarkan classification report, maka dihasilkan nilai sebagai berikut:

| Model | Label | Precision | Accuracy |
|-------|-------|-----------|----------|
| K-Nearest Neighbors | 0| 0.71 | 0.70 |
| | 1 | 0.69 | |
| Random Forest | 0 | 0.95 | 0.96 |
| | 1 | 0.97 | |
| CatBoost | 0 | 0.96 | 0.97 |
| | 1 | 0.98 | |

Dari hasil pemodelan, dapat disimpulkan bahwa model CatBoost memiliki performa yang paling baik dibandingkan dengan model K-Nearest Neighbors dan Random Forest. CatBoost juga merupakan model yang fleksibel dan dapat menangani data kategorikal tanpa perlu encoding. Model tersebut melakukan konversi data kategorikal menjadi data numerik secara otomatis menggunakan algoritma novel yang membantu dalam mengurangi overfitting. Selain itu, model CatBoost mengimplementasikan mekanisme boosting, yaitu teknik mengatasi masalah overfitting dengan mengacak urutan data pada setiap iterasi pelatihan, sehingga model tidak akan terlalu fokus pada data yang sama[2]. Hal ini membuat model CatBoost cocok untuk digunakan dalam memprediksi kepuasan penumpang dengan tepat meskipun data yang diperoleh sangat besar dan beragam.

![important_feature](https://github.com/Dzoel31/Project-Predictive-Analytics-Dicoding/blob/main/assets/important_feature.png?raw=true)

Merujuk kepada gambar di atas, fitur-fitur yang menjadi penentu kepuasan penumpang adalah *Inflight wifi service*, *Type of Travel*, *Online Boarding*, dan *Customer Type*. Perusahaan maskapai dapat memperhatikan fitur-fitur tersebut untuk mempertahankan atau meningkatkan kepuasan pelanggan. Selain itu, perusahaan dapat memberikan perbaikan pada fitur-fitur lain yang memiliki kontribusi rendah terhadap kepuasan penumpang.

Perusahaan dapat mengambil langkah-langkah berikut untuk meningkatkan kualitas layanan:

1. Maskapai dapat meningkatkan kualitas pada layanan yang memiliki kontribusi yang signifikan. Misalnya pada layanan *Inflight wifi service*, maskapai dapat meningkatkan kualitas layanan wifi dengan meningkatkan kecepatan internet dan ketersediaan wifi selama penerbangan.
2. Maskapai dapat memberikan layanan yang lebih baik pada pelanggan yang melakukan perjalanan bisnis. Misalnya dengan memberikan fasilitas khusus untuk pelanggan bisnis, seperti ruang tunggu khusus, prioritas boarding, dan lain sebagainya.
3. Maskapai dapat meningkatkan kualitas layanan online boarding dengan mengimplementasikan teknologi yang lebih mutakhir untuk mengurangi waktu tunggu.
4. Maskapai dapat memperbaiki layanan-layanan yang memiliki kontribusi rendah terhadap kepuasan penumpang. Misalnya layanan *Seat Comfort*, maskapai dapat meningkatkan kenyamanan kursi dengan memberikan kursi yang lebih empuk dan lega.

## Daftar Pustaka

[1] "puas". Kamus Besar Bahasa Indonesia Daring. Jakarta: Badan Pengembangan dan Pembinaan Bahasa. 2016.

[2] Kolli, A. (2024, February 13). [Understanding CatBoost: The gradient boosting algorithm for categorical data](https://aravindkolli.medium.com/understanding-catboost-the-gradient-boosting-algorithm-for-categorical-data-73ddb200895d). Medium.

[3] Sasongko, S. R. (2021). Faktor-faktor kepuasan pelanggan dan loyalitas pelanggan (literature review manajemen pemasaran). Jurnal ilmu manajemen terapan, 3(1), 104-114.

[4] Harjati, L., & Venesia, Y. (2015). [Pengaruh kualitas layanan dan persepsi harga terhadap kepuasan pelanggan pada maskapai penerbangan Tiger Air Mandala](http://download.garuda.kemdikbud.go.id/article.php?article=400299&val=6685&title=PENGARUH%20KUALITAS%20LAYANAN%20DAN%20PERSEPSI%20HARGA%20TERHADAP%20KEPUASAN%20PELANGGAN%20PADA%20MASKAPAI%20PENERBANGAN%20TIGER%20AIR%20MANDALA). E-Journal Widya Ekonomika, 1(1), 36791.
