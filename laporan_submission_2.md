# Laporan Proyek Machine Learning - Nida'an Khafiyya

## Project Overview

Seiring dengan meningkatnya jumlah buku digital yang tersedia di berbagai platform, pengguna semakin sulit menemukan bacaan yang sesuai dengan preferensi mereka. Hal ini menciptakan fenomena *information overload* yang menghambat akses pengguna terhadap materi yang relevan dan bermanfaat, terutama di lingkungan pendidikan tinggi. Sistem rekomendasi muncul sebagai solusi yang efektif untuk membantu pengguna menavigasi koleksi besar dengan menyajikan saran yang personal dan relevan. Namun, tantangan utama dalam sistem ini adalah bagaimana memahami preferensi pengguna dan karakteristik buku secara efisien, khususnya ketika data eksplisit seperti rating atau ulasan terbatas. Maka dari itu, pengembangan sistem rekomendasi yang adaptif menjadi penting untuk mengatasi masalah tersebut dan meningkatkan keterlibatan pengguna.

Penelitian sebelumnya oleh Ahmed dan Letta menunjukkan bahwa pendekatan berbasis *collaborative filtering* seperti matrix factorization (SVD) mampu menangani permasalahan klasik sistem rekomendasi, seperti *cold start* dan *data sparsity*, dengan hasil akurasi yang menjanjikan (RMSE 0.1623). Studi tersebut juga menyoroti kelemahan pendekatan konvensional seperti *association rule mining* dan *case-based reasoning* yang menghasilkan daftar item yang statis dan kurang responsif terhadap kebutuhan pengguna baru [[1]](https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801). Oleh karena itu, penting untuk mengeksplorasi penggabungan pendekatan konten dan kolaboratif guna menghasilkan rekomendasi yang lebih kontekstual, adaptif, dan mampu memberikan pengalaman yang lebih personal kepada pengguna, khususnya dalam domain perpustakaan digital pendidikan.

Referensi: [Book Recommendation Using Collaborative Filtering Algorithm](https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah:
- Bagaimana cara membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, terutama ketika jumlah koleksi buku sangat besar dan membuat proses pencarian menjadi kurang efisien?
- Bagaimana cara membangun sistem rekomendasi yang lebih personal dan relevan, yang mampu mempertimbangkan pola interaksi serta preferensi individual pengguna secara lebih mendalam?
- Bagaimana cara mengatasi cold-start problem, yaitu kondisi di mana pengguna baru atau buku baru belum memiliki riwayat interaksi atau rating, sehingga tetap memungkinkan sistem memberikan rekomendasi yang akurat?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan sistem rekomendasi berbasis content-based filtering yang dapat merekomendasikan lima buku serupa berdasarkan informasi penulis bukudan penerbit. Sistem ini akan bekerja dengan mengukur kesamaan antar buku untuk membantu pengguna mengeksplorasi bacaan yang sejenis.
- Membangun sistem rekomendasi berbasis collaborative filtering yang mampu merekomendasikan sepuluh buku baru kepada pengguna berdasarkan histori interaksi pengguna dengan buku-buku sebelumnya. Sistem ini akan memanfaatkan data rating untuk mempelajari pola preferensi pengguna.
- Mengatasi masalah cold-start pada pengguna atau buku baru dengan mengandalkan pendekatan berbasis konten, sehingga tetap memungkinkan sistem memberikan rekomendasi meskipun belum tersedia data eksplisit seperti rating.

### Solution statements

- Content-Based Filtering: Menggunakan pendekatan TF-IDF untuk merepresentasikan metadata buku seperti nama penulis dan nama penerbit dalam bentuk vektor numerik, menghitung kemiripan antar buku menggunakan cosine similarity untuk merekomendasikan lima buku yang mirip dengan buku yang disukai pengguna, dan sangat efektif dalam mengatasi kasus cold-start karena tidak bergantung pada data interaksi pengguna.
- Collaborative Filtering dengan RecommenderNet: Menerapkan RecommenderNet, model deep learning yang mempelajari representasi laten dari pengguna dan buku berdasarkan data interaksi berupa rating, untuk memprediksi skor relevansi buku yang belum dibaca dan merekomendasikan 10 buku dengan skor tertinggi, sehingga mampu memberikan rekomendasi yang personal dan akurat berdasarkan pola perilaku pengguna lain yang serupa.

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari **Kaggle**, dengan judul **Book Recommendation Dataset**. Dataset ini terdiri dari tiga tabel utama, yaitu `Books`, `Ratings`, dan `Users`, yang masing-masing disimpan dalam format CSV. Dataset ini dikumpulkan dari interaksi pengguna dengan buku di platform penjualan **Amazon Web Services**.
Adapun rincian data adalah sebagai berikut:

* **Books.csv** terdiri dari **271.360 baris data** dan **8 kolom fitur** yang berisi informasi detail mengenai buku.
* **Ratings.csv** terdiri dari **1.149.780 baris data** dan **3 kolom fitur** yang merepresentasikan penilaian pengguna terhadap buku.
* **Users.csv** terdiri dari **278.858 baris data** dan **3 kolom fitur** mengenai informasi pengguna.

Sumber Dataset: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

### **Deskripsi Variabel**

**Books.csv**

* `ISBN` : Nomor identifikasi unik untuk setiap buku.
* `Book-Title` : Judul buku.
* `Book-Author` : Nama penulis buku.
* `Year-Of-Publication` : Tahun terbit buku.
* `Publisher` : Nama penerbit buku.
* `Image-URL-S` : URL gambar buku ukuran kecil.
* `Image-URL-M` : URL gambar buku ukuran sedang.
* `Image-URL-L` : URL gambar buku ukuran besar.

**Ratings.csv**

* `User-ID` : ID unik pengguna.
* `ISBN` : Nomor ISBN buku yang diberi rating.
* `Book-Rating` : Skor rating yang diberikan pengguna (rentang 0–10).

**Users.csv**

* `User-ID` : ID unik pengguna.
* `Location` : Lokasi tempat tinggal pengguna.
* `Age` : Usia pengguna.

### **Exploratory Data Analysis (Univariate)**
1. **Dataset: Books**

Dataset `books` berisi informasi mengenai buku-buku yang tersedia di dalam sistem Book-Crossing. Tujuan awal dari eksplorasi ini adalah untuk memahami karakteristik data buku secara umum.

Berikut adalah beberapa informasi statistik awal yang diperoleh:

* Jumlah total entri buku (ISBN): 271.360
* Jumlah judul buku unik (`Book-Title`): 242.135
* Jumlah penulis unik (`Book-Author`): 102.023
* Jumlah penerbit unik (`Publisher`): 16.808
* Jumlah nilai unik pada kolom tahun terbit (`Year-Of-Publication`): 202

**Perbedaan Jumlah ISBN dan Judul Buku**

Ditemukan perbedaan antara jumlah ISBN dan judul buku (Book-Title). Oleh karena itu, kode di bawah ini digunakan untuk melakukan pengecekan jumlah kemunculan setiap judul buku:
```
books['Book-Title'].value_counts()
```
Hasil dari kode tersebut menunjukkan bahwa terdapat beberapa buku dengan judul yang sama. 
![image](https://github.com/user-attachments/assets/9764d074-d0cd-4ca1-b439-9a22affd8e5d)

Sebagai contoh, pada judul "Selected Poems", pencarian dengan judul yang sama menghasilkan beberapa entri buku yang identik pada judul, tetapi berbeda pada ISBN, penulis, penerbit, atau tahun terbit.
| ISBN       | Book-Title     | Book-Author             | Year-Of-Publication | Publisher                             |
| ---------- | -------------- | ----------------------- | ------------------- | ------------------------------------- |
| 081120958X | Selected Poems | William Carlos Williams | 1985                | New Directions Publishing Corporation |
| 0811201465 | Selected Poems | K. Patchen              | 1957                | New Directions Publishing Corporation |
| 0679750800 | Selected Poems | Rita Dove               | 1993                | Vintage Books USA                     |

**Pembersihan Kolom Tahun Terbit**

![image](https://github.com/user-attachments/assets/fa91a3a1-17e5-4a3b-8f0e-4c820fb73f63)

Ketika melakukan eksplorasi terhadap kolom `Year-Of-Publication`, ditemukan bahwa beberapa entri berisi data yang bukan merupakan tahun, seperti nama penerbit (`DK Publishing Inc`, `Gallimard`) atau angka yang tidak logis (misalnya, `0`, `1376`, `2020`, `2050`).

Setelah dilakukan investigasi lebih lanjut, kasus-kasus seperti ini ternyata terjadi karena pergeseran data antar kolom saat proses input. Untuk menjaga integritas data, tiga entri dengan kesalahan input tersebut dihapus.

Kemudian untuk nilai-nilai tahun yang tidak realistis (di luar rentang wajar publikasi, seperti `0` atau di atas `2006`) diubah menjadi nilai kosong (`NaN`) dan diisi dengan rata-rata tahun yang valid menggunakan codingan di bawah ini.

```
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.nan

# mengganti NaN dengan nilai rata-rata yearOfPublication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].mean()), inplace=True)
```
Setelah tahapan tersebut, kolom tahun terbit telah berisi nilai yang realistis. Kemudian, tipe data pada kolom `Year-Of-Publication` diubah dari tipe data `object` menjadi `integer` agar bisa diproses secara numerik.
![image](https://github.com/user-attachments/assets/36e6c91d-e4c5-4f0f-83ab-3a5409e35a9d)

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `books`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| Book-Author         | 2              |
| Publisher           | 2              |
| ISBN                | 0              |
| Book-Title          | 0              |
| Year-Of-Publication | 0              |

Hanya ada dua kolom yang memiliki *missing values*, dan jumlahnya sangat kecil sehingga tidak signifikan terhadap keseluruhan dataset.

---

2. **Dataset: Ratings**

Dataset `ratings` mencatat penilaian yang diberikan oleh pengguna terhadap buku. Dataset ini menjadi sangat penting karena akan digunakan untuk analisis preferensi pengguna dan sistem rekomendasi.

Statistik dasar yang ditemukan:

* Jumlah pengguna yang memberikan rating: 105.283
* Jumlah total rating yang tercatat: 340.556
* Jumlah skala rating unik: 11

**Skala Rating**

Rating diberikan dalam skala **0 hingga 10**, dengan total 11 nilai unik. Berikut adalah interpretasi awal:

* **Rating 0** kemungkinan besar berarti tidak ada rating eksplisit yang diberikan oleh pengguna. Ini dikenal sebagai *implicit rating* atau bisa jadi kesalahan input.
* Rating 1–10 menunjukkan penilaian eksplisit, dengan nilai lebih tinggi menandakan tingkat kesukaan yang lebih besar.

Distribusi nilai rating:

| Skor Rating | Keterangan              |
| ----------- | ----------------------- |
| 0           | Tidak diberi / default  |
| 1–3         | Penilaian sangat rendah |
| 4–6         | Penilaian sedang        |
| 7–10        | Penilaian tinggi        |

**Statistik Deskriptif Rating**

Hasil dari `ratings.describe()` memberikan ringkasan sebagai berikut:

* **Count**: 1.149.780
* **Mean**: 2.866
* **Std (standar deviasi)**: 3.854
* **Min**: 0
* **25%**: 0
* **50% (median)**: 0
* **75%**: 8
* **Max**: 10

Nilai tengah (median) berada di 0, mengindikasikan bahwa sebagian besar rating tidak eksplisit (banyak 0). Namun, nilai kuartil atas (75%) adalah 8, yang memperlihatkan bahwa ketika rating diberikan, nilainya cenderung tinggi.

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `ratings`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| User-ID         | 0              |
| ISBN           | 0              |
| Book-Rating                | 0              |

Tidak ada missing value dalam dataset ini.

---

3. **Dataset: Users**

Dataset `users` mencatat informasi dasar tentang pengguna, seperti ID, lokasi, dan umur.

Informasi awal yang diperoleh:

* Jumlah user unik: 278.858
* Jumlah lokasi unik (`Location`): 57.339
* Jumlah nilai unik pada kolom `Age`: 166

**Umur Pengguna**

![image](https://github.com/user-attachments/assets/1ec0355e-bfef-4fa7-9b2f-e5557d08938c)

Kolom `Age` memiliki nilai-nilai yang tidak masuk akal seperti `0`, `231`, dan `244`. Maka dilakukan pembersihan dengan cara:

* Umur di bawah 5 tahun atau di atas 90 tahun dianggap tidak valid dan diubah menjadi `NaN`
* Nilai `NaN` kemudian diisi dengan rata-rata umur pengguna yang valid
* Tipe data diubah ke `integer` agar lebih konsisten

Berikut code yang digunakan
```
# Umur di bawah 5 dan di atas 90 tidak masuk akal, maka menggantinya dengan NaN
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan

# Replacing NaNs with mean
users.Age = users.Age.fillna(users.Age.mean())

# Setting the data type as int
users.Age = users.Age.astype(np.int32)
```

Hasil deskriptif umur pengguna setelah dibersihkan:

| Statistik | Nilai |
| --------- | ----- |
| Rata-rata | 34.4  |
| Median    | 34    |
| Minimum   | 5     |
| Maksimum  | 90    |

Sebagian besar pengguna berada di rentang usia produktif (sekitar 29–35 tahun), menunjukkan demografi utama pengguna platform ini.

**Missing Values**

Berikut adalah tabel jumlah *missing values* (nilai kosong) dalam dataset `users`:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| User-ID         | 0              |
| Location           | 0              |
| Age                | 0              |

Tidak ada missing value dalam dataset ini.

### Visualisasi data
1. **Books**

**Barplot 10 Author Teratas dengan Buku Terbitan Terbanyak**
![image](https://github.com/user-attachments/assets/5e920802-dc0c-40c5-bdf2-2e1640a0ba2f)
Insight:
Berdasarkan barplot yang menampilkan 10 penulis teratas dengan jumlah buku terbitan terbanyak, terlihat jelas bahwa Agatha Christie mendominasi dengan jumlah publikasi yang signifikan, jauh melampaui penulis lainnya. William Shakespeare menempati posisi kedua, diikuti oleh Stephen King di urutan ketiga. Secara keseluruhan, grafik ini memperlihatkan distribusi jumlah buku yang diterbitkan oleh para penulis terkemuka, dengan penurunan bertahap dari penulis dengan publikasi terbanyak hingga penulis di urutan kesepuluh, yaitu Charles Dickens.

**Barplot 10 Publisher Teratas dengan Buku Terbitan Terbanyak**
![image](https://github.com/user-attachments/assets/b1e14994-2f40-44e3-868d-8d151713c194)
Insight:
Barplot ini menyajikan informasi mengenai 10 penerbit teratas berdasarkan jumlah buku yang telah mereka terbitkan. Terlihat bahwa Harlequin menduduki posisi puncak dengan jumlah publikasi yang jauh lebih tinggi dibandingkan penerbit lainnya. Silhouette berada di urutan kedua, diikuti oleh Pocket dan Ballantine Books yang memiliki jumlah publikasi serupa. Secara keseluruhan, grafik ini menggambarkan adanya dominasi yang signifikan dari beberapa penerbit besar dalam industri penerbitan buku, dengan penurunan jumlah publikasi secara bertahap hingga penerbit di urutan kesepuluh, yaitu Warner Books.

2. **Ratings**

**Barplot Pembagian Pemeringkatan Buku**
![image](https://github.com/user-attachments/assets/ccaecce4-7fba-4c7d-89b6-a3150d58b37e)
Insight:
Barplot ini memperlihatkan distribusi peringkat buku, dengan jelas menunjukkan bahwa mayoritas besar peringkat terkonsentrasi pada nilai 0. Hal ini mengindikasikan bahwa terdapat sejumlah besar buku yang belum atau tidak mendapatkan peringkat. Setelah nilai 0, jumlah peringkat secara umum meningkat seiring dengan kenaikan nilai peringkat, mencapai puncaknya pada peringkat 8, kemudian sedikit menurun pada peringkat 9 dan 10. Distribusi ini menyiratkan adanya polarisasi dalam pemberian peringkat, di mana banyak buku tidak dinilai sama sekali, sementara di antara buku yang dinilai, terdapat kecenderungan pemberian peringkat yang lebih tinggi.

3. **Users**

**Barplot Distribusi Umur Pengguna**
![image](https://github.com/user-attachments/assets/fae6290b-d808-44ad-85a9-7ecb7cb44f76)
Insight:
Boxplot ini menyajikan distribusi usia pengguna. Terlihat bahwa sebagian besar data usia terkumpul di sekitar rentang usia 25 hingga 45 tahun, yang ditunjukkan oleh kotak utama. Garis hitam di dalam kotak merepresentasikan median usia. Ekor atau *whisker* pada kedua sisi kotak menunjukkan sebaran data usia di luar kuartil bawah dan atas. Titik di atas *whisker* atas mengindikasikan adanya nilai *outlier*, yaitu satu pengguna dengan usia yang jauh lebih tinggi dibandingkan dengan mayoritas pengguna lainnya.

## Data Preparation

Pada tahap ini, dilakukan berbagai langkah pembersihan dan transformasi data untuk memastikan data yang digunakan bersih, relevan, dan siap untuk proses analisis lebih lanjut maupun pembangunan model sistem rekomendasi. Tahapan yang dilakukan adalah sebagai berikut:

### Melihat Ukuran Awal Dataset Buku
```
print('Banyak buku: ', len(books['ISBN'].unique()))
```
Kode `print('Banyak buku: ', len(books['ISBN'].unique()))` digunakan untuk menampilkan jumlah total buku unik yang terdapat dalam dataset `books` berdasarkan kolom 'ISBN'. Informasi ini memberikan gambaran mengenai skala dataset buku secara keseluruhan.

### Reduksi Ukuran Dataset
```
# Mengambil 15.000 baris data secara acak dari DataFrame books dengan random_state=5
reduced_books = books.sample(n=15000, random_state=5)
reduced_books

reduced_ratings = ratings[ratings['ISBN'].isin(unique_isbn_list)]
reduced_ratings
```
Untuk mempermudah dan mempercepat proses analisis serta pembangunan model rekomendasi, dilakukan reduksi ukuran dataset `books` dan `ratings`. Untuk mengurangi ukuran dataset `books`, diambil sampel sebanyak 15.000 baris data secara acak menggunakan fungsi `books.sample(n=15000, random_state=5)`. Parameter `random_state=5` digunakan untuk memastikan hasil pengambilan sampel dapat direproduksi.

### Pengambilan Sampel Data Buku
```
unique_isbn_list = reduced_books['ISBN'].unique().tolist()
len(unique_isbn_list)
```
*Dataframe* hasil pengambilan sampel ini disimpan dalam variabel `reduced_books`. Jumlah ISBN unik dalam sampel ini kemudian diperiksa menggunakan kode `unique_isbn_list = reduced_books['ISBN'].unique().tolist()` dan `len(unique_isbn_list)`.

### Pemfilteran Data Rating
```
reduced_ratings = ratings[ratings['ISBN'].isin(unique_isbn_list)]
reduced_ratings
```
*Dataframe* `ratings` difilter untuk hanya menyertakan rating buku-buku yang ISBN-nya terdapat dalam daftar ISBN unik dari `reduced_books`. Hal ini dilakukan menggunakan kode `reduced_ratings = ratings[ratings['ISBN'].isin(unique_isbn_list)]`. Dengan demikian, *dataframe* `reduced_ratings` hanya berisi rating untuk buku-buku yang juga terdapat dalam sampel `reduced_books`.

### Penghapusan Kolom URL Gambar
```
reduced_books = reduced_books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
```
Tiga kolom yang berisi informasi URL gambar buku, yaitu 'Image-URL-S', 'Image-URL-M', dan 'Image-URL-L', dihapus dari *dataframe* `reduced_books` menggunakan kode `reduced_books = reduced_books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])`. Kolom-kolom ini dianggap tidak relevan untuk analisis rekomendasi berbasis teks atau rating.

### Penggabungan Data Rating dan Buku
```
# Menggabungkan dataframe rating dengan book berdasarkan nilai ISBN
booksrate = pd.merge(reduced_ratings, reduced_books, on='ISBN', how='left')
booksrate
```
*Dataframe* `reduced_ratings` dan `reduced_books` digabungkan berdasarkan kolom 'ISBN' menggunakan fungsi `pd.merge(reduced_ratings, reduced_books, on='ISBN', how='left')`. Penggunaan `how='left'` memastikan bahwa semua rating dari `reduced_ratings` tetap ada, dan informasi buku yang sesuai dari `reduced_books` ditambahkan. Hasil penggabungan disimpan dalam *dataframe* `booksrate`.

### Penghitungan Jumlah Rating per Buku
```
# Menghitung jumlah rating kemudian menggabungkannya berdasarkan ISBN
booksrate.groupby('ISBN').sum()
```
Kode `booksrate.groupby('ISBN').sum()` digunakan untuk menghitung jumlah total rating untuk setiap ISBN dalam *dataframe* `booksrate`. Meskipun menggunakan fungsi `sum()`, yang relevan di sini sebenarnya adalah ukuran dari setiap grup ISBN, yang mengindikasikan berapa kali setiap buku dinilai. Hasil ini memberikan informasi mengenai popularitas atau frekuensi rating setiap buku dalam subset data.

### Penghapusan Missing Value

Setelah dilakukan proses EDA, ditemukan bahwa masih terdapat beberapa nilai kosong (*missing values*) dalam gabungan dataset `booksrate`. Berikut ini adalah jumlah missing value dari masing-masing kolom:

| Kolom               | Missing Values |
| ------------------- | -------------- |
| User-ID             | 0              |
| ISBN                | 0              |
| Book-Rating         | 0              |
| Book-Title          | 0              |
| Book-Author         | 0              |
| Year-Of-Publication | 0              |
| Publisher           | 1              |

Terlihat bahwa hanya ada satu *missing value* pada kolom `Publisher`, dan sisanya sudah bersih. Untuk membersihkan data dari *missing value*, digunakan fungsi `dropna()`:

```python
booksrate_clean = booksrate.dropna()
```

Fungsi ini secara otomatis menghapus baris yang memiliki nilai kosong pada salah satu kolom. Karena hanya ada satu baris yang terpengaruh, maka tidak akan berdampak signifikan terhadap keseluruhan dataset. Dengan menghapus missing value maka dapat menghindari error pada saat melakukan analisis atau pemodelan dan memastikan integritas data tetap terjaga tanpa baris yang tidak lengkap.

### Penghapusan Duplikat Data

Setelah membersihkan *missing value*, langkah selanjutnya adalah memastikan tidak ada data duplikat dalam data buku. Duplikat yang dimaksud di sini adalah data dengan ISBN yang sama muncul lebih dari sekali.

```python
preparation = booksrate_clean.drop_duplicates('ISBN')
```

Kolom `ISBN` (International Standard Book Number) merupakan pengenal unik untuk setiap buku. Dengan menghapus data duplikat berdasarkan kolom ini, kita menjamin bahwa tidak ada entri buku yang redundan dalam data yang akan diproses lebih lanjut. Dengan langkah ini, maka dapat menghindari bias data karena satu buku yang sama muncul lebih dari sekali dan menjaga akurasi dalam sistem rekomendasi yang berbasis konten (content-based filtering), karena sistem akan mengenali setiap buku hanya satu kali.

### Mengonversi Kolom Menjadi List

Setelah data dibersihkan dari duplikat, kolom-kolom penting seperti `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, dan `Publisher` kemudian dikonversi ke dalam bentuk list.

```python
book_isbn = preparation['ISBN'].tolist()
book_title = preparation['Book-Title'].tolist()
book_author = preparation['Book-Author'].tolist()
book_year = preparation['Year-Of-Publication'].tolist()
book_publisher = preparation['Publisher'].tolist()
```

Semua list ini memiliki panjang yang sama yaitu 14.930, menandakan bahwa tidak ada data yang hilang dalam proses konversi.Dengan melakukan langkah ini, maka dapat mengubah data menjadi bentuk list memudahkan manipulasi data pada tahap pembuatan *content-based recommender* dan format list lebih fleksibel untuk digunakan dalam pembuatan *dictionary*, *TF-IDF vectorizer*, atau pembuatan fitur metadata gabungan.

### Data Preparation untuk Content-Based Filtering

1. **Pembuatan Data `books_new`**

Langkah awal adalah menyusun *dataframe* baru bernama `books_new`. 
```
# Membuat dictionary untuk data ‘book_isbn’, ‘book_title’, ‘book_author’, dan 'book_publisher'
books_new = pd.DataFrame({
    'isbn': book_isbn,
    'title': book_title,
    'author': book_author,
    'year': book_year,
    'publisher': book_publisher
})
books_new
```

Proses ini melibatkan konversi informasi buku dari format *series* (kemungkinan dari hasil pembacaan *dataset* sebelumnya) menjadi format *list*. *List* ini kemudian digunakan untuk membuat kolom-kolom dalam *dataframe* `books_new`, yang terdiri dari 'isbn', 'book_title', 'book_author', 'year_of_publication', dan 'publisher'. Tujuan dari pembentukan *dataframe* ini adalah untuk memudahkan manipulasi dan transformasi data ke dalam format yang sesuai untuk pemodelan *content-based filtering*.

2. **Penggabungan Fitur Teks**

Untuk merepresentasikan konten setiap buku, informasi dari kolom 'book_author' dan 'publisher' digabungkan menjadi satu kolom teks baru bernama 'combined'. 
```
# Gabungkan author dan publisher menjadi satu kolom string
data['combined'] = data['author'].fillna('') + ' ' + data['publisher'].fillna('')
```

Langkah ini dilakukan dengan mengisi nilai yang hilang (jika ada) pada kedua kolom tersebut dengan string kosong ('') dan kemudian menggabungkannya dengan spasi di antara keduanya. Kolom 'combined' ini akan menjadi dasar untuk perhitungan kemiripan konten antar buku.

**3. Ekstraksi Fitur dengan TF-IDF**

Teknik *Term Frequency-Inverse Document Frequency* (TF-IDF) digunakan untuk mengekstrak fitur-fitur penting dari teks dalam kolom 'combined'.

* **Inisialisasi `TfidfVectorizer()`**
  ```
  # Inisialisasi TF-IDF
  tf = TfidfVectorizer()
  ```
  Sebuah objek `TfidfVectorizer` dibuat. *Vectorizer* ini akan mengubah teks menjadi matriks representasi numerik, di mana setiap kata dalam korpus akan menjadi sebuah fitur. Bobot TF-IDF diberikan kepada setiap kata dalam setiap dokumen (dalam hal ini, setiap entri di kolom 'combined'), yang mencerminkan seberapa penting kata tersebut dalam dokumen relatif terhadap seluruh korpus.
* **Fit dan Transform ke Kolom 'combined'**
  ```
  # Fit dan transform ke kolom gabungan
  tfidf_matrix = tf.fit_transform(data['combined'])
  ```
  Metode `fit_transform()` dari objek `TfidfVectorizer` dipanggil dengan kolom 'combined' sebagai input. Proses `fit` akan mempelajari kosakata dari seluruh teks dalam kolom 'combined', dan proses `transform` akan mengubah setiap entri teks menjadi vektor TF-IDF berdasarkan kosakata yang telah dipelajari. Hasilnya adalah matriks `tfidf_matrix`.
* **Melihat Fitur yang Dihasilkan**
  ```
  # Melihat fitur yang dihasilkan
  features = tf.get_feature_names_out()
  
  # print(tfidf_matrix.shape)
  tf.get_feature_names_out()
  ```
  `tf.get_feature_names_out()` digunakan untuk mendapatkan daftar semua fitur (kata unik) yang telah diekstrak oleh `TfidfVectorizer`.
* **Melihat Ukuran Matriks TF-IDF**
  ```
  # Melakukan fit lalu ditransformasikan ke bentuk matrix
  tfidf_matrix = tf.fit_transform(data['combined'])
  
  # Melihat ukuran matrix tfidf
  tfidf_matrix.shape
  ```
  `tfidf_matrix.shape` memberikan dimensi dari matriks TF-IDF. Jumlah baris sesuai dengan jumlah buku dalam *dataset*, dan jumlah kolom sesuai dengan jumlah fitur unik yang ditemukan dalam kolom 'combined'.
* **Mengubah ke Matriks Padat**
  ```
  # Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
  tfidf_matrix.todense()
  ```
  `tfidf_matrix.todense()` mengubah matriks *sparse* TF-IDF menjadi matriks padat. Meskipun representasi *sparse* lebih efisien untuk penyimpanan dan perhitungan dengan data teks berdimensi tinggi, representasi padat mungkin lebih mudah dipahami dalam beberapa kasus.
* **Membuat *Dataframe* Matriks TF-IDF**
  ```
  # Membuat dataframe untuk melihat tf-idf matrix
  # Kolom diisi dengan author, publisher
  # Baris diisi dengan nama buku
  
  pd.DataFrame(
      tfidf_matrix.todense(),
      columns=tf.get_feature_names_out(),
      index=data.title
  ).sample(22, axis=1).sample(10, axis=0)
  ```
  Sebuah *dataframe* dibuat dari matriks padat TF-IDF. Kolom-kolom *dataframe* ini diberi nama sesuai dengan fitur-fitur yang diekstrak (kata-kata dari 'author' dan 'publisher'), dan indeksnya adalah judul buku dari kolom 'title' (kemungkinan kolom 'book_title' telah diubah namanya menjadi 'title' pada tahap sebelumnya). Metode `.sample()` digunakan untuk menampilkan sebagian kecil dari *dataframe* ini, memudahkan visualisasi bobot TF-IDF untuk beberapa buku dan fitur secara acak.

**Tujuan Keseluruhan**

Tujuan dari tahapan persiapan data ini adalah untuk mengubah representasi tekstual dari penulis dan penerbit buku menjadi representasi numerik (vektor TF-IDF) dan kemudian mengukur kemiripan antar buku berdasarkan representasi numerik ini menggunakan *cosine similarity*. Matriks *cosine similarity* yang dihasilkan (`cosine_sim_df`) akan menjadi dasar untuk sistem rekomendasi *content-based*, di mana buku-buku dengan nilai kemiripan *cosine* yang tinggi dianggap memiliki konten yang serupa dan dapat direkomendasikan satu sama lain.

### Data Preparation untuk Collaborative Filtering

1. **Penggunaan *Dataframe* `reduced_ratings`:**
```
# Membaca dataset
df = reduced_ratings
df
```
Langkah awal dalam persiapan data untuk *collaborative filtering* adalah menggunakan *dataframe* yang disebut `reduced_ratings`. Ini mengindikasikan bahwa mungkin telah dilakukan proses pengurangan atau pemilihan sebagian data rating dari *dataset* awal. Tujuannya bisa untuk mengurangi kompleksitas komputasi atau fokus pada interaksi pengguna dan item yang lebih relevan.

2. **Encoding User ID dan ISBN**

Proses *encoding* adalah mengubah data kategorikal (dalam hal ini, ID pengguna dan ISBN buku yang kemungkinan besar berupa *string* atau angka unik) menjadi representasi numerik berupa indeks integer. Hal ini penting karena model *machine learning*, termasuk model *collaborative filtering*, umumnya bekerja lebih baik dengan input numerik.

* **Membuat *List* ID Unik:** Kode `df['User-ID'].unique().tolist()` dan `df['ISBN'].unique().tolist()` menghasilkan *list* yang berisi nilai-nilai unik dari kolom 'User-ID' dan 'ISBN'. Ini memastikan bahwa setiap pengguna dan setiap buku hanya direpresentasikan satu kali dalam *list*.

* **Membuat *Dictionary* Pemetaan (Encoding):** Dua *dictionary* dibuat untuk setiap ID pengguna dan ISBN buku:
    * `user_to_user_encoded` dan `book_to_book_encoded`
        ```
        # Melakukan encoding userID
        user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
        print('encoded userID : ', user_to_user_encoded)
    
        # Melakukan proses encoding ISBN
        book_to_book_encoded = {x: i for i, x in enumerate(book_isbn)}
        ```
      *Dictionary* ini memetakan setiap nilai unik (ID pengguna atau ISBN) ke sebuah indeks integer yang berurutan. Misalnya, pengguna dengan ID '22' mungkin dipetakan ke indeks 0, pengguna dengan ID '53' ke indeks 1, dan seterusnya.
    * `user_encoded_to_user` dan `book_encoded_to_book`
        ```
        # Melakukan proses encoding angka ke ke userID
        user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
        print('encoded angka ke userID: ', user_encoded_to_user)
    
        # Melakukan proses encoding angka ke ISBN
        book_encoded_to_book = {i: x for i, x in enumerate(book_isbn)}
        ```
      Ini adalah *dictionary* kebalikan dari yang sebelumnya, memetakan indeks integer kembali ke nilai asli (ID pengguna atau ISBN). Ini berguna untuk interpretasi hasil model.

* **Menerapkan Pemetaan ke *Dataframe***
  ```
  # Mapping User-ID ke dataframe user
  df['user'] = df['User-ID'].map(user_to_user_encoded)
    
  # Mapping ISBN ke dataframe book
  df['book'] = df['ISBN'].map(book_to_book_encoded)
  ```
  Metode `.map()` digunakan untuk menerapkan *dictionary* *encoding* ke kolom 'User-ID' dan 'ISBN' dalam *dataframe* `df`. Ini akan membuat dua kolom baru, 'user' dan 'book', yang berisi indeks integer yang sesuai untuk setiap interaksi (rating).

**3. Mendapatkan Jumlah Pengguna dan Buku:**
```
# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah buku
num_book = len(book_to_book_encoded)
print(num_book)
```
Setelah proses *encoding*, jumlah pengguna unik (`num_users`) dan jumlah buku unik (`num_book`) dihitung menggunakan panjang dari *dictionary* *encoding*. Informasi ini penting untuk menentukan dimensi dari *embedding layer* dalam model *neural network* untuk *collaborative filtering*.

**4. Konversi Tipe Data Rating:**
```
# Mengubah rating menjadi nilai float
df['Book-Rating'] = df['Book-Rating'].values.astype(np.float32)
```
Kolom 'Book-Rating' diubah menjadi tipe data *float32*. Ini umum dilakukan untuk nilai *rating* karena model *machine learning* seringkali bekerja dengan bilangan *floating-point*.

**5. Identifikasi Rentang Rating:**
```
# Nilai minimum rating
min_rating = min(df['Book-Rating'])

# Nilai maksimal rating
max_rating = max(df['Book-Rating'])
```
Nilai minimum (`min_rating`) dan maksimum (`max_rating`) dari kolom 'Book-Rating' diidentifikasi. Informasi ini akan digunakan untuk normalisasi nilai *rating* ke dalam rentang yang lebih kecil (biasanya 0 hingga 1).

**6. Pengacakan Dataset:**
```
# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df
```
*Dataframe* `df` diacak menggunakan `df.sample(frac=1, random_state=42)`. `frac=1` berarti semua baris akan dikembalikan, tetapi dalam urutan acak. `random_state=42` digunakan untuk memastikan bahwa pengacakan akan menghasilkan urutan yang sama setiap kali kode dijalankan, yang penting untuk *reproducibility*.

**7. Pemisahan Fitur dan Label:**
```
# Membuat variabel x untuk mencocokkan data user dan book menjadi satu value
x = df[['user', 'book']].values

# Membuat variabel y untuk membuat rating dari hasil
y = df['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```
* **Fitur (`x`):** Kolom 'user' dan 'book' (yang berisi indeks *encoding*) dipilih sebagai fitur (`x`). Pasangan indeks pengguna dan buku ini akan menjadi input untuk model *collaborative filtering*. Metode `.values` digunakan untuk mengubah *dataframe* menjadi *array numpy*.
* **Label (`y`):** Kolom 'Book-Rating' digunakan sebagai label (`y`). Nilai *rating* dinormalisasi ke dalam rentang 0 hingga 1 menggunakan formula: `(x - min_rating) / (max_rating - min_rating)`. Normalisasi ini membantu model untuk belajar dengan lebih stabil dan efisien.

**8. Pembagian Data Latih dan Validasi:**
```
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```
Dataset dibagi menjadi set pelatihan (80%) dan validasi (20%). `train_indices` dihitung untuk menentukan titik pemisahan. Kemudian, fitur (`x`) dan label (`y`) dibagi menjadi `x_train`, `x_val`, `y_train`, dan `y_val`. Set pelatihan akan digunakan untuk melatih model, sedangkan set validasi akan digunakan untuk mengevaluasi kinerja model selama pelatihan dan membantu dalam *tuning hyperparameter*.

**Tujuan Keseluruhan**

Secara keseluruhan, tahapan ini mempersiapkan data *rating* pengguna dan buku ke dalam format numerik yang sesuai untuk melatih model *collaborative filtering*. Proses *encoding* mengubah identitas pengguna dan buku menjadi indeks, normalisasi skala *rating*, dan pembagian data memastikan bahwa model dapat dipelajari dan dievaluasi dengan baik.

## Modeling

Pada tahap ini, dua pendekatan algoritma yang berbeda diimplementasikan untuk membangun sistem rekomendasi buku: *Content-Based Filtering* (CBF) dan *Collaborative Filtering* (CF). Setiap pendekatan memiliki cara kerja, kelebihan, dan kekurangan yang berbeda dalam menghasilkan rekomendasi.

### Content-Based Filtering

Pendekatan *Content-Based Filtering* merekomendasikan buku kepada pengguna berdasarkan kemiripan atribut konten antar buku. Dalam implementasi ini, fitur 'author' dan 'publisher' dari buku digunakan sebagai dasar untuk mengukur kemiripan.

**Cara Kerja:**

1.  **Representasi Fitur:** Fitur teks dari kolom 'combined' (gabungan 'author' dan 'publisher') diubah menjadi vektor numerik menggunakan algoritma *Term Frequency-Inverse Document Frequency* (TF-IDF). TF-IDF memberikan bobot pada setiap kata berdasarkan frekuensinya dalam satu buku dan invers frekuensi dokumennya di seluruh koleksi buku.
2.  **Perhitungan Kemiripan:** Kemiripan antar vektor TF-IDF dari setiap buku dihitung menggunakan fungsi `cosine_similarity()` dari *library* `sklearn.metrics.pairwise`. *Cosine similarity* adalah ukuran kemiripan antara dua vektor dalam ruang multidimensi, dengan nilai yang berkisar antara -1 (tidak mirip) hingga 1 (sangat mirip). Hasil perhitungan ini disimpan dalam matriks *numpy* (`cosine_sim`).
3.  **Membuat Matriks Kemiripan (*Cosine Similarity*) dalam Bentuk *Dataframe*:** Matriks *numpy* `cosine_sim` kemudian diubah menjadi *dataframe* `cosine_sim_df`. Judul buku dari kolom 'title' digunakan sebagai indeks dan nama kolom pada *dataframe* ini. Setiap sel *(i, j)* dalam `cosine_sim_df` menunjukkan nilai *cosine similarity* antara buku *i* dan buku *j*. Ukuran dari *dataframe* ini adalah persegi dengan dimensi jumlah buku dikali jumlah buku.
4.  **Penyusunan Rekomendasi:** Untuk memberikan rekomendasi untuk sebuah buku yang dipilih, sistem mencari buku-buku lain dengan skor *cosine similarity* tertinggi terhadap buku tersebut dalam `cosine_sim_df`. Buku dengan kemiripan tertinggi dianggap paling relevan.

**Implementasi Fungsi `book_recommendations`:**
```python
def book_recommendations(title, similarity_data=cosine_sim_df, items=data[['title', 'author', 'publisher']], k=5):
  index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))

  # Mengambil data dengan similarity terbesar dari index yang ada
  closest = similarity_data.columns[index[-1:-(k+2):-1]]

  # Drop title agar nama buku yang dicari tidak muncul dalam daftar rekomendasi
  closest = closest.drop(title, errors='ignore')

  return pd.DataFrame(closest).merge(items).head(k)
```
Fungsi `book_recommendations` menerima judul buku (`title`), *dataframe* kemiripan (`similarity_data`), *dataframe* informasi buku (`items`), dan jumlah rekomendasi yang diinginkan (`k`) sebagai input. Fungsi ini bekerja dengan:

1.  Mencari indeks buku yang sesuai dengan judul yang diberikan dalam *dataframe* kemiripan.
2.  Menggunakan `argpartition` untuk mendapatkan indeks dari *k* buku yang paling mirip (berdasarkan skor *cosine similarity* tertinggi).
3.  Mengambil nama-nama buku yang paling mirip dari kolom *dataframe* kemiripan berdasarkan indeks yang ditemukan.
4.  Menghapus judul buku yang menjadi input dari daftar rekomendasi agar tidak muncul sebagai rekomendasi.
5.  Menggabungkan daftar judul buku rekomendasi dengan *dataframe* informasi buku untuk menampilkan detail penulis dan penerbit.
6.  Mengembalikan *dataframe* berisi *k* buku rekomendasi teratas.

**Contoh Hasil Rekomendasi CBF:**

Untuk buku "Harry Potter and the Prisoner of Azkaban (Book 3)", sistem merekomendasikan 5 buku teratas berdasarkan kemiripan penulis dan penerbit:

```
                                                title         author                 publisher
0  Harry Potter and the Chamber of Secrets (Harry...       J. K. Rowling       Arthur A. Levine Books
1    Harry Potter and the Philosopher's Stone (Cove...     J.K. Rowling        BBC Consumer Publishing
2  Exploring Space: From Ancient Legends to the T...      Scholastic Books           Scholastic
3                   Harry Potter und der Stein der Weisen Joanne K. Rowling      Carlsen Verlag GmbH
4  Harry Potter und der Gefangene von Askaban. So...      Joanne K. Rowling       Dhv der HÃ¶rverlag
```

**Kelebihan CBF:**

* **Tidak Membutuhkan Data Pengguna:** Rekomendasi didasarkan sepenuhnya pada atribut item, sehingga tidak memerlukan riwayat interaksi pengguna.
* **Transparansi:** Alasan di balik rekomendasi relatif mudah dipahami (berdasarkan kemiripan fitur).
* **Mampu Merekomendasikan Item Baru:** Dapat merekomendasikan buku baru yang memiliki fitur serupa dengan buku yang disukai pengguna di masa lalu.

**Kekurangan CBF:**

* **Keterbatasan Fitur:** Kualitas rekomendasi sangat bergantung pada kualitas dan kelengkapan fitur item.
* **Over-Specialization:** Cenderung merekomendasikan item yang sangat mirip dengan preferensi masa lalu pengguna, sehingga kurang dalam penemuan (*discovery*) item baru yang mungkin menarik.
* **Tidak Mempertimbangkan Preferensi Pengguna Lain:** Tidak memanfaatkan informasi dari pengguna lain yang memiliki selera serupa.

### Collaborative Filtering

Pendekatan *Collaborative Filtering* merekomendasikan buku kepada pengguna berdasarkan pola interaksi (rating) pengguna lain yang memiliki preferensi serupa. Dalam implementasi ini, digunakan model *neural network* dengan arsitektur `RecommenderNet`.

**Cara Kerja:**

1.  **Pembuatan Matriks Interaksi:** Data rating pengguna dan buku diubah menjadi matriks interaksi pengguna-buku, di mana baris mewakili pengguna, kolom mewakili buku, dan nilai sel menunjukkan rating yang diberikan pengguna untuk buku tersebut (jika ada).
2.  **Pembelajaran Representasi Laten (Embedding):** Model `RecommenderNet` menggunakan *embedding layer* untuk mempelajari representasi laten (vektor berdimensi rendah) untuk setiap pengguna dan setiap buku. *Embedding* ini menangkap fitur-fitur tersembunyi yang mendasari preferensi pengguna dan karakteristik buku berdasarkan pola interaksi.
3.  **Prediksi Rating:** Model memprediksi rating yang mungkin diberikan seorang pengguna untuk buku yang belum pernah ia interaksikan, berdasarkan *embedding* pengguna dan buku.
4.  **Penyusunan Rekomendasi:** Buku-buku dengan prediksi rating tertinggi yang belum pernah dibaca oleh pengguna direkomendasikan.

**Implementasi Model `RecommenderNet`:**
```
class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_book, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_book = num_book
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.book_embedding = layers.Embedding( # layer embeddings book
        num_book,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.book_bias = layers.Embedding(num_book, 1) # layer embedding book bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    book_vector = self.book_embedding(inputs[:, 1]) # memanggil layer embedding 3
    book_bias = self.book_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_book = tf.tensordot(user_vector, book_vector, 2)

    x = dot_user_book + user_bias + book_bias

    return tf.nn.sigmoid(x) # activation sigmoid
```
Model `RecommenderNet` adalah kelas *Keras Model* yang terdiri dari:

* **User Embedding Layer:** Memetakan setiap indeks pengguna ke vektor *embedding* berdimensi `embedding_size`.
* **User Bias Layer:** Mempelajari bias spesifik untuk setiap pengguna.
* **Book Embedding Layer:** Memetakan setiap indeks buku ke vektor *embedding* berdimensi `embedding_size`.
* **Book Bias Layer:** Mempelajari bias spesifik untuk setiap buku.

Fungsi `call` dalam model melakukan operasi *dot product* antara vektor *embedding* pengguna dan buku, kemudian menambahkan bias pengguna dan buku. Hasilnya diaktifkan menggunakan fungsi sigmoid untuk menghasilkan prediksi rating dalam skala 0 hingga 1 (karena rating telah dinormalisasi).

Model dikompilasi menggunakan fungsi *loss* `BinaryCrossentropy` (meskipun ini adalah masalah regresi rating, *binary cross-entropy* sering digunakan dalam *implicit feedback* atau setelah mengubah rating menjadi sinyal biner), *optimizer* Adam, dan metrik evaluasi *Root Mean Squared Error* (RMSE). Model dilatih menggunakan data pelatihan (`x_train`, `y_train`) dan dievaluasi pada data validasi (`x_val`, `y_val`).

**Contoh Hasil Rekomendasi CF:**

Untuk seorang pengguna dengan ID 11676, sistem merekomendasikan 10 buku teratas yang belum pernah dibaca oleh pengguna tersebut, berdasarkan prediksi rating model:

```
Showing recommendations for users: 11676
===========================
Book with high ratings from user
--------------------------------
Shadowrun. Deutschland in den Schatten. : Hans Joachim Alpers - Heyne
The Professor and the Madman: A Tale of Murder, Insanity, and the Making of The Oxford English Dictionary : Simon Winchester - Perennial
Wolfwalker : Tara K. Harper - Del Rey Books
When Strangers Marry : Lisa Kleypas - Avon Books
Think on These Things : J. Krishnamurti - HarperCollins Publishers
Our Town: A Play in Three Acts : Thornton Niven Wilder - Harpercollins
Opposites (First Concepts) : Melanie Whittington - Priddy Books
Hawk O'Toole's Hostage : Sandra Brown - Bantam
Manuel Alvarez Bravo (Phaidon 55's) : Amanda Hopkinson - Phaidon Press
Exploring Natural Disasters (Eyes on Adventure Series) : Stella Sands - Kidsbooks Inc
--------------------------------
Top 10 books recommendation
--------------------------------
The Baby Book: Everything You Need to Know About Your Baby from Birth to Age Two : Martha Sears - Little, Brown
Warchild : Karin Lowachee - Aspect
The scarlet letter (The World's best reading) : Nathaniel Hawthorne - Reader's Digest Association
Life Is So Good : George Dawson - Penguin Books
This Little Light of Mine: The Life of Fannie Lou Hamer : Kay Mills - Plume Books
The Chronicles of Pern: First Fall (The Dragonriders of Pern) : Anne McCaffrey - Ballantine Books
Le Combat ordinaire, tome 1 : Larcenet - Dargaud
Blitzeis. : Peter Stamm - btb
Keeping Watch : LAURIE R. KING - Bantam
The School Story : Andrew Clements - Aladdin
```

**Kelebihan CF:**

* **Rekomendasi Personal:** Dapat memberikan rekomendasi yang sangat personal karena didasarkan pada preferensi pengguna lain yang serupa.
* **Penemuan Item Baru:** Mampu merekomendasikan item yang mungkin tidak terkait secara konten tetapi disukai oleh pengguna dengan selera yang sama.
* **Tidak Membutuhkan Fitur Item Eksplisit:** Bekerja dengan baik bahkan jika informasi fitur item terbatas.

**Kekurangan CF:**

* **Masalah *Cold Start*:** Sulit memberikan rekomendasi untuk pengguna baru tanpa riwayat interaksi atau item baru tanpa interaksi apa pun.
* **Sparsitas Data:** Kinerja dapat menurun jika matriks interaksi pengguna-item sangat jarang (banyak pengguna belum berinteraksi dengan banyak item).
* **Skalabilitas:** Dengan jumlah pengguna dan item yang besar, komputasi dapat menjadi mahal.

Dalam implementasi ini, kedua pendekatan memberikan jenis rekomendasi yang berbeda. CBF merekomendasikan buku yang mirip dalam hal penulis dan penerbit, sementara CF merekomendasikan buku berdasarkan pola rating pengguna lain yang memiliki preferensi serupa. Kombinasi kedua pendekatan (*hybrid recommendation system*) seringkali dapat memberikan hasil rekomendasi yang lebih baik dengan memanfaatkan kelebihan masing-masing metode dan mengatasi beberapa kekurangannya.

### **Perbandingan Singkat**

| Aspek                | Content-Based Filtering             | Collaborative Filtering          |
| -------------------- | ----------------------------------- | -------------------------------- |
| Berdasarkan          | Fitur item                          | Interaksi pengguna-item          |
| Cold-start problem   | Tidak untuk pengguna, ya untuk item | Ya, jika tidak ada data pengguna |
| Eksplorasi item baru | Terbatas (mirip saja)               | Bisa beragam                     |
| Data yang dibutuhkan | Metadata buku                       | Riwayat interaksi pengguna       |

## Evaluation

Bagian ini akan membahas metrik evaluasi yang digunakan untuk mengukur kinerja kedua pendekatan sistem rekomendasi, *Content-Based Filtering* (CBF) dan *Collaborative Filtering* (CF), serta menganalisis hasil proyek berdasarkan metrik tersebut.

### Evaluasi Content-Based Filtering

Untuk mengevaluasi kinerja sistem rekomendasi berbasis konten, digunakan metrik **Precision**, **Recall**, dan **F1-Score**. Metrik ini umum digunakan dalam tugas klasifikasi dan relevan untuk mengevaluasi apakah buku-buku yang direkomendasikan memang mirip dengan buku yang menjadi dasar rekomendasi.

* **Precision:** Mengukur proporsi buku yang relevan di antara semua buku yang direkomendasikan. Cara kerja Precision adalah dengan menghitung dari semua buku yang direkomendasikan oleh sistem, berapa proporsi di antaranya yang sebenarnya relevan dengan preferensi pengguna (berdasarkan ground truth kemiripan yang ditetapkan). Jika Precision tinggi, berarti sistem sangat akurat dalam merekomendasikan buku yang benar-benar mirip. Secara matematis, Precision didefinisikan sebagai:

  $$\text{Precision} = \frac{\text{Jumlah Buku Relevan yang Direkomendasikan}}{\text{Total Jumlah Buku yang Direkomendasikan}}$$

  Precision yang tinggi menunjukkan bahwa ketika sistem merekomendasikan sebuah buku, kemungkinan besar buku tersebut memang relevan.

* **Recall:** Mengukur proporsi buku relevan yang berhasil direkomendasikan oleh sistem dari semua buku yang sebenarnya relevan. Cara kerja Recall adalah dengan menghitung dari semua buku yang sebenarnya relevan dengan preferensi pengguna (berdasarkan ground truth kemiripan), berapa proporsi di antaranya yang berhasil direkomendasikan oleh sistem. Jika Recall tinggi, berarti sistem mampu menemukan sebagian besar buku yang relevan. Secara matematis, Recall didefinisikan sebagai:

  $$\text{Recall} = \frac{\text{Jumlah Buku Relevan yang Direkomendasikan}}{\text{Total Jumlah Buku yang Sebenarnya Relevan}}$$

  Recall yang tinggi menunjukkan bahwa sistem mampu mengidentifikasi sebagian besar buku yang relevan.

* **F1-Score:** Merupakan rata-rata harmonik antara Precision dan Recall. Metrik ini memberikan gambaran yang seimbang mengenai performa model, terutama ketika terdapat ketidakseimbangan antara Precision dan Recall. Cara kerja F1-Score adalah dengan mengambil rata-rata harmonik antara Precision dan Recall. Ini memberikan ukuran tunggal yang menyeimbangkan kemampuan sistem untuk menjadi akurat (Precision) dan lengkap (Recall). F1-Score sangat berguna ketika kita ingin menghindari sistem yang hanya fokus pada salah satu aspek (misalnya, merekomendasikan sangat sedikit buku tetapi semuanya relevan, atau merekomendasikan banyak buku tetapi hanya sedikit yang relevan). Secara matematis, F1-Score didefinisikan sebagai:

  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Dalam implementasi, kemiripan *cosine* antara buku digunakan sebagai dasar untuk menentukan relevansi. Sebuah *threshold* kemiripan ditetapkan (dalam kasus ini 0.6). Jika kemiripan antara dua buku melebihi *threshold*, buku tersebut dianggap relevan. *Ground truth* dibuat berdasarkan *threshold* ini, dan prediksi biner dibuat berdasarkan apakah skor kemiripan melebihi *threshold*.

**Hasil Evaluasi CBF:**

```
Precision: 1.0000
Recall:    1.0000
F1-score:  1.0000
```

Hasil evaluasi menunjukkan bahwa dengan *threshold* 0.6, model *Content-Based Filtering* mencapai skor Precision, Recall, dan F1-Score sempurna (1.0). Ini mengindikasikan bahwa semua buku yang dianggap mirip oleh model (di atas *threshold*) memang dianggap relevan berdasarkan *ground truth* yang dibuat, dan model berhasil mengidentifikasi semua pasangan buku yang relevan dalam sampel yang diuji. Namun, perlu diingat bahwa evaluasi ini dilakukan pada sampel data dan sangat bergantung pada *threshold* yang dipilih. Skor sempurna ini mungkin tidak berlaku untuk seluruh dataset atau dengan *threshold* yang berbeda.

### Evaluasi Collaborative Filtering

Untuk mengevaluasi kinerja model *Collaborative Filtering*, digunakan metrik **Root Mean Squared Error (RMSE)**. RMSE adalah metrik umum untuk mengevaluasi model regresi, dan dalam konteks sistem rekomendasi, RMSE mengukur perbedaan antara rating yang diprediksi oleh model dan rating sebenarnya yang diberikan oleh pengguna. Semakin kecil nilai RMSE, semakin akurat prediksi rating model.

* **Root Mean Squared Error (RMSE):** Dihitung sebagai akar kuadrat dari rata-rata kuadrat perbedaan antara nilai prediksi ($\hat{y}_i$) dan nilai sebenarnya ($y_i$). Secara matematis, RMSE didefinisikan sebagai:

  ![image](https://github.com/user-attachments/assets/810738e2-dedd-435c-ae4a-092065632190)
  
    di mana $n$ adalah jumlah total prediksi.
  Cara kerja RMSE adalah dengan mengukur rata-rata besarnya kesalahan prediksi rating yang dilakukan oleh model. Untuk setiap interaksi pengguna-buku dalam data validasi, model memprediksi rating. RMSE kemudian menghitung selisih antara rating prediksi dan rating sebenarnya, mengkuadratkan selisih ini (untuk menghilangkan nilai negatif dan memberikan bobot lebih besar pada kesalahan yang lebih besar), mencari rata-rata dari semua selisih kuadrat, dan akhirnya mengambil akar kuadrat dari rata-rata tersebut. Hasilnya adalah ukuran kesalahan rata-rata dalam skala rating asli (atau skala normalisasi dalam kasus ini). RMSE yang rendah menunjukkan bahwa model secara keseluruhan membuat prediksi rating yang dekat dengan rating sebenarnya.

**Hasil Evaluasi CF:**

Berdasarkan grafik metrik model yang ditampilkan, terlihat bahwa nilai RMSE pada data pelatihan terus menurun seiring dengan bertambahnya *epoch*. Nilai RMSE pada data validasi juga menurun pada awalnya, namun kemudian cenderung stabil atau bahkan sedikit meningkat setelah *epoch* tertentu.

Nilai RMSE terakhir pada *epoch* ke-20 adalah:

* **RMSE Training:** Sekitar 0.2728
* **RMSE Validasi:** Sekitar 0.3605

Berikut visualisasi model metrik.
![image](https://github.com/user-attachments/assets/ec5ab25f-afd9-4cfe-a126-5f7ad088f822)


Perbedaan antara RMSE pelatihan dan validasi menunjukkan adanya sedikit *overfitting*, di mana model belajar terlalu baik pada data pelatihan sehingga kinerjanya sedikit menurun pada data yang belum pernah dilihat (data validasi). Namun, nilai RMSE validasi sebesar 0.3605 menunjukkan bahwa model masih memiliki kemampuan yang cukup baik dalam memprediksi rating buku oleh pengguna. Semakin rendah nilai RMSE, semakin akurat prediksi rating model. Dalam konteks rating yang dinormalisasi antara 0 dan 1, RMSE sebesar 0.3605 menunjukkan bahwa rata-rata kesalahan prediksi rating adalah sekitar 0.36 pada skala yang dinormalisasi.

### Evaluasi Tujuan Proyek Berdasarkan Problem Statements

Berdasarkan hasil evaluasi:

* **Bagaimana cara membantu pengguna menemukan buku yang sesuai dengan preferensi mereka?** Kedua model berkontribusi. CBF merekomendasikan berdasarkan kemiripan konten (penulis, penerbit), membantu pengguna menemukan buku serupa dengan yang mereka sukai. CF merekomendasikan berdasarkan pola rating pengguna lain, memperluas penemuan berdasarkan preferensi kolektif.

* **Bagaimana cara membangun sistem rekomendasi yang lebih personal dan relevan?** CF secara khusus bertujuan untuk ini dengan mempelajari *embedding* pengguna dan buku dari data interaksi, menghasilkan rekomendasi yang dipersonalisasi. CBF juga memberikan rekomendasi yang relevan berdasarkan konten buku yang diminati pengguna.

* **Bagaimana cara mengatasi *cold-start problem*?** CBF memiliki keunggulan dalam mengatasi *cold-start* untuk buku baru karena rekomendasinya hanya bergantung pada atribut konten. CF lebih rentan terhadap *cold-start* karena memerlukan data interaksi pengguna dan buku.

Secara keseluruhan, kedua pendekatan berhasil diimplementasikan dan dievaluasi. Content-Based Filtering efektif dalam merekomendasikan buku yang kontennya mirip, sementara Collaborative Filtering mampu memberikan rekomendasi yang dipersonalisasi berdasarkan pola interaksi pengguna. Pemilihan pendekatan yang paling sesuai atau kombinasi keduanya dapat bergantung pada kebutuhan spesifik dan karakteristik dataset yang lebih luas.

**------**

_Catatan:_
- Ahmed, E., & Letta, A. (2023). Book Recommendation Using Collaborative Filtering Algorithm. Applied Computational Intelligence and Soft Computing, 2023, Article ID 1514801. https://onlinelibrary.wiley.com/doi/10.1155/2023/1514801
- Kaggle. (n.d.). Book Recommendation Dataset. https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
- Dicoding. (n.d.). Machine Learning Terapan. https://www.dicoding.com/academies/319-machine-learning-terapan
