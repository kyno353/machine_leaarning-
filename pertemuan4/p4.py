import pandas as pd

# Baca CSV
df = pd.read_csv(r"pertemuan4/kelulusan_mahasiswa.csv")


# Info dataset
print(df.info())
print(df.head())



# Cek missing value
print(df.isnull().sum())

# Hapus duplikat (kalau ada)
df = df.drop_duplicates()

# Deteksi outlier dengan boxplot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.boxplot(x=df['IPK'])
plt.show()

df['IPK'] = df['IPK'].fillna(df['IPK'].median())

# Statistik deskriptif
print(df.describe())

# Histogram IPK
plt.figure(figsize=(6,4))
sns.histplot(df['IPK'], bins=10, kde=True)
plt.show()

# Scatterplot
plt.figure(figsize=(6,4))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.show()

# Heatmap korelasi
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

# Simpan ke file baru
df.to_csv("processed_kelulusan.csv", index=False)

from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Train 70%, sisanya 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Dari 30% dibagi 2 (validation 15%, test 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)