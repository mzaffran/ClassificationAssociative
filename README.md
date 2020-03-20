# Classification Associative

_Auteurs : Walid Makhlouf et Margaux Zaffran_

Projet pour le cours SOD322 : Recherche opérationnelle et données massives, partie de Zacharie Ales.

## Dépendances

### Julia

- JuMP (version 21.1)
- CPLEX
- DataFrames
- CSV

### Python

- pandas
- sklearn
- matplotlib

### R

- corrplot
- randomForest
- rpart

## Mode opérationnel

Afin de reproduire les  résultats présentés dans le notebook, il faut :

__ORC__

Modifier le fichier ```main.jl``` pour :
- adapter la variable ```respectProp``` (_true_ si on veut imposer le respect des proportions des classes lors de la séparation train/test, _false_ sinon) ;
- mettre les variables ```deleteData```, ```deleteCreate``` et ```deleteSort``` à _true_ pour réaliser de nouvelles expériences, sinon l'algorithme utilisera les fichiers déjà présents dans les dossiers ```data``` et ```res``` ;
- pour lancer ORC avec notre choix de binarisation, mettre ```dataSet = kidney``` ;
- pour lancer ORC avec toutes les variables binarisées selon leurs histogrammes, mettre ```dataSet = kidneyAll``` ;
- pour lancer ORC avec toutes les variables binarisées naïvement, mettre ```dataSet = kidneyAllGreedy``` ;
- pour lancer un ORC multiclasses sur la variable SG, mettre ```dataSet = multiclass``` (si vous avez nettoyé le dossier ```data``` ou simplement supprimé le fichier ```data/multiclass.csv``` alors il faut au préalable exécuter le script python ```create_multiclass.py```).;

Executer ensuite ```include("src/main.jl")``` dans un terminal Julia.

__ORC bi-objectif__

Modifier le fichier ```mainMultiObj.jl``` pour :
- adapter la variable ```respectProp``` (_true_ si on veut imposer le respect des proportions des classes lors de la séparation train/test, _false_ sinon) ;
- mettre les variables ```deleteData```, ```deleteCreate``` et ```deleteSort``` à _true_ pour réaliser de nouvelles expériences, sinon l'algorithme utilisera les fichiers déjà présents dans les dossiers ```data``` et ```res```, et mettre ```dataSet = kidneyMultiObj``` ;
- pour lancer un nouvel ORC bi-objectif avec les train et test générés lors d'une exécution précédente d'ORC classique, mettre ```dataSet = kidney```, ```deleteData = false```, ```deleteCreate = false``` et ```deleteSort = false``` ;

Executer ensuite ```include("src/mainMultiObj.jl")``` dans un terminal Julia.

__Méthodes ML__

Pour comparer ces algorithmes aux méthodes CART et random forest, les codes sont disponibles dans ```ml_methods.R```. Ne pas oublier de modifier la troisième ligne ```setwd``` en mettant le bon chemin vers le bon répertoire.

Pour comparer ces algorithmes aux méthodes SVC et AdaBoost, les codes sont disponibles dans ```ml_methods.py```.

Ces codes récupèrent par défaut les train et test nommés ```kidney_train.csv``` et ```kidney_test.csv``` dans ```data```. En principe, c'est donc les jeux de train et test générés par l'ORC auquel nous souhaitons nous comparer.
