from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #classe che permete di fare tutti i calcoli
from sklearn.neural_network import MLPClassifier
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
    mask = (y == i)
    plt.scatter(X[mask, 0], X[mask,1], label=iris.target_names[i])
plt.xlabel('Lunghezza sepalo')
plt.ylabel('Larghezza sepalo')
plt.title('Distribuzione dei Sepali')
plt.legend()



# Plot per i sepali
plt.subplot(1, 2, 2)
for i in range(3):
    mask = (y == i)
    plt.scatter(X[mask, 2], X[mask,3], label=iris.target_names[i])
plt.xlabel('Lunghezza petalo')
plt.ylabel('Larghezza petalo')
plt.title('Distribuzione dei petali')
plt.legend()

plt.show()
# Plot per i petali

# Ora tocca a te! Prova a fare il plot dei petali!
# Poi fai commit e push

#fiz acima, agora vamos treinar 
#divido i dati in train e test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=90) #la aleno poco e lo stresso nel test

scaler = StandardScaler()#vado a creare una variabile di tipo standard scaler
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
print(f"minimo: {min(X_train_scaled[:,0])}, massimo: {max(X_train_scaled[:,0])}, media: {np.mean (X_train_scaled[:,0])}")
#calcoliamo i dati di test
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes = (4,2), #4 neuroni,2 layer
    activation="tanh",
    random_state=99, #questo randomiza i pesi
    max_iter=10000
)
mlp.fit(X_train_scaled, y_train)
y_predict = mlp.predict(X_test_scaled)
accuracy = np.mean(y_predict == y_test) #mean = media
print(f"Accuratezza: {accuracy:.2f}")
print(f"Test loss: {mlp.loss_}")
print(f"Numero iterazioni: {mlp.n_iter_}")
#fino ora abiamo alenato la rete, adesso dobbiamo utilizarla
nuovo_iris = [(5.0, 3.5, 1.5, 0.2)]#lungheza dei sepali e largheza, matrice= righe sono osservazione, colone sono le fiture
#predict va utilizare la rete neurale
nuovo_iris_scaled = scaler.transform(nuovo_iris) #va scalare i dati come nei dati di training
previsione_iris = mlp.predict(nuovo_iris_scaled)
print("Il mio nuovo fiore è.....")
print("...Ruollo di tamburi...")
print(iris.target_names[previsione_iris[0]])
#non ho il modello matematico, ma abbiamo il modello neurale