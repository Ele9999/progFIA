# Progetto Fondamenti di Intelligenza Artificiale
L’obiettivo del progetto consiste nell’analizzare dati che provengono da 79 pazienti anonimizzati dai quali sono state acquisite immagini TAC del distretto anatomico dell’addome.

A tutti i 79 pazienti è stata inizialmente diagnosticata una neoplasia maligna alla prostata e si sono sottoposti ad una prostatectomia radicale.

Successivamente, durante un esame di “controllo” nel quale sono state raccolte le TAC in analisi, in 45 dei pazienti è stata riscontrata una recidiva del tumore.

Si chiede di sviluppare un sistema in IA che, utilizzando il dato acquisito in questa visita di controllo, predica la presenza di recidiva del tumore.

**Il task consiste quindi nel predire l’etichetta binaria Recidiva/Non_Recidiva per ogni paziente**

Dentro alla folder **data-istruzioni** sono presenti i seguenti file:
* *Data.xlsx* -> dataset composto da 79 righe e 251 colonne (ai fini della computazione ignoriamo la prima riga e la prima colonna)
* *Istruzioni.pdf* -> file in cui sono presenti i dettagli del progetto

## Descrizione del progetto

Per svolgere al meglio il progetto si è effettuato un **pre-processing** sui dati, andando a eliminare le colonne irrilevanti ai fini computazionali e andando a sostituire le celle vuote (NaN) del dataset con la media.

Per la **features selection** è stato utilizzato *SelectKBest* con la funzione f_classif, che valuta la relazione tra ogni feature e la variabile target (nel nostro caso, Recidiva/Non Recidiva) usando l'analisi della varianza

Come **modello di machine learning** si è optato per il *Random Forest*

Infine, si è eseguita una valutazione del modello attraverso le seguenti **metriche di valutazione**:
* *Accuratezza*
* *Precisione*
* *Recall*
* *f1*

Come **metodo di validazione**, invece, si è optato per il *cross-validation*

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

3. Installare tutte le librerie necessarie presenti all'interno del file *requirements.txt*

```bash
pip install -r requirements.txt'
```

4. Eseguire, rispettando l'ordine, le celle presenti all'interno del file *Progetto_ok.ipynb* se si desidera vedere punto punto il codice

5. Altrimenti, eseguire il codice *Progetto_ok.py*

```bash
python3 Progetto_ok.py
```





