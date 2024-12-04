from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Caricamento del dataset
iris = load_iris()
print(iris)
X = iris.data
y = iris.target

# Visualizzazione della distribuzione delle classi
plt.figure(figsize=(12, 4))

# Plot per i sepali
plt.subplot(1, 2, 1)
for i in range(3):
    mask = (y == i) # Y sono i risultari, y==i prendi tutti i valori della specie che correspondo il vetore y, mask: qualle valore per arrey Y, corrispondo a la specie 1,2,3 
    plt.scatter(X[mask, 0], X[mask,1], label=iris.target_names[i]) #
plt.xlabel('Lunghezza sepalo')
plt.ylabel('Larghezza sepalo')
plt.title('Distribuzione dei Sepali')
plt.legend()

# Plot per i petali

# Ora tocca a te! Prova a fare il plot dei petali!
# Poi fai commit e push

plt.show()