## Table of Contents
1. [General Info](#general-info)
2. [Utilisation du code de l'application](#application)
3. [Utilisation du code de calcul de l'indice cubital](#calcul)

### General Info
***
Dans ce fichier vous allez pouvoir trouver des codes qui réalisent le traitement des ailes, prises sur une plaque quadrillée.
Les parties découpe et filtrage de l'aile sont réalisés à travers les codes de l'application. Il suffit de la lancer pour exécuter ce traitement.
La suite de la réalisation du projet n'est pas relié à cette application et nécessite d'exécuter ce code à part.

## Utilisation du code de l'application
***
Afin de lancer l'application, se positionner sur votre IDE Python et ouvrir un projet. Sélectionner le dossier appli, ce sera le projet.
Ensuite, lancer l'exécution du main et tout ce déroulera en lisant les instructions sur l'interface.
ATTENTION : à la ligne 96 du main est demandée une adresse.
```
$ aile_filtrage(filename, tab_im_final, '/Users/chloepoulic/Documents/BZZZZ_final/stockage')
```
Il faut changer celle indiquée et mettre le chemin du dossier dans lequel vous souhaitez récupérer les données filtrées.

## Utilisation du code de calcul de l'indice cubital
***
Dans ce code, il est nécessaire de rentrer le numéro de l'aile et le numéro de la ruche dont on souhaite étudier l'individu. Le code importera alors l'image du token souhaité, précédemment découpé et enregistré. (faire attention aux adresses et techniques d'importation)
En sortie, ce code retourne l'indice cubital calculé et la conclusion tirée à partir de celui ci sur l'espèce identifiée.

## Utilisation du code de reconnaissance et extraction de token : "phaseCorrelation.py" 
*** 
le code "PhaseCorrelationRotation.py" est une ébauche de code, essayant de placer toutes les ailes traitées suivant une même orientation. Ce code n'est pas nécessaire car les apiculteurs placent désormais les ailes dans des cases et cela empêche les ailes d'être d'orientation hétérogène.

Au début du code "phaseCorrelation.py", il vous est demandé de donner le numéro de la ruche dont vous voulez extraire les motifs d'interêt, le nombre d'ailes de cette ruche et le nombre de tokens de la base de données "tokens de référence".
Une fois cela fait, exécuter le code (run) et 5 figures apparaîtrons (il s'agit de l'extraction du motif de la dernière aile) et vous pouvez vous rendre, pendant que l'algorithme tourne, dans le dossier "token_trouve" de la ruche étudié, pour voir apparaître les motifs extraits.
Plusieurs chemins peuvent être à modifier, suivant le dossier dans lequel vous vous placerez : 
- ligne 75-76 
- ligne 94-95
- ligne 168-169 => choix de la base de tokens de référence à choisir 
- ligne 216 
