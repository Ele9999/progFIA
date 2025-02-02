# Progetto Fondamenti di Intelligenza Artificiale
L’obiettivo del progetto consiste nell’analizzare dati che provengono da 79 pazienti dai quali sono state acquisite immagini TAC del distretto anatomico dell’addome. A tutti i 79 pazienti è stata inizialmente diagnosticata una neoplasia maligna alla prostata e si sono sottoposti ad una prostatectomia radicale. Successivamente, durante un esame di “controllo” nel quale sono state raccolte le TAC in analisi, in 45 dei pazienti è stata riscontrata una recidiva del tumore.

Si chiede di sviluppare un sistema in IA che, utilizzando il dato acquisito in questa visita di controllo, predica la presenza di recidiva del tumore.

**Il task consiste quindi nel predire l’etichetta binaria Recidiva/Non_Recidiva per ogni paziente**

Dentro alla folder **data-istruzioni** sono presenti i seguenti file:
* *Data.xlsx* -> dataset composto da 79 righe e 251 colonne (ai fini della computazione ignoriamo la prima riga e la prima colonna)
* *Istruzioni.pdf* -> file in cui sono presenti i dettagli del progetto

## Descrizione del progetto

Per svolgere al meglio il progetto si è effettuato un **data cleaning** sui dati, andando a eliminare le colonne irrilevanti ai fini computazionali e andando a sostituire le celle vuote (NaN) del dataset con la media.

Per quanto riguarda la **features selection**, non è stata necessaria dato che l'algoritmo implementato è efficiente dataset ad alta dimensionalità.

Come **modello di machine learning** si è optato per il *Support Vector Machine (SVC)* dopo averlo confrontato con il *Random Forest*

Infine, si è eseguita una valutazione del modello attraverso le seguenti **metriche di valutazione**:
* *Accuratezza*
* *Precisione*
* *Recall*
* *f1-score*

Per osservare meglio le performance si è utilizzata la **matrice di confusione** e la **curva ROC**

## Implementazione del codice

Per poter implementare il codice seguire i seguenti step:

1. Assicurarsi di avere python installato

```bash
python3 --version
```

2. Clonare il repository in locale

```bash
git clone https://github.com/Ele9999/progFIA.git
```

3. Installare tutti i moduli necessari presenti all'interno del file *requirements.txt*

```bash
pip install -r requirements.txt
```

4. Se si vuole visualizzare il codice passo passo, vedere *SVC.ipynb* (qui sono presenti anche dei grafici)

5. Se si vuole solo eseguire il codice e vedere i risultati:

```bash
python3 Progetto.py
```