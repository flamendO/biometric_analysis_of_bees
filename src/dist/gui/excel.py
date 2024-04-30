import pandas as pd
import numpy as np
import os 
import openpyxl

"""

Fonction qui prends un dataframe en entrée et un chemin de fichier, la structre du dataframe doit être comme suit :
N lignes correspondants aux N images analysées, 2 colonnes pour les valeurs des indices. Elle ne retourne rien mais
sauvegarde un fichier contenant les valeurs des indices, les moyennes et écarts types.

"""

def toExcel(dataset, file_path):
    path = os.getcwd()
    filename = "Apiculteur.xlsx"
    travail = openpyxl.Workbook()
    sheet = travail.active
    sheet.title = "Données"
    lettre_debut = 65
    
    print("\nNous sommes à la feuille : " + str(sheet.title))
   
    # Écriture des noms de colonnes, pour se placer sur une case du fichier Excel il faut écrire 
    # LETTRENUMERO, par exemple : A1. L'indicage commence à 1 
    for j in range(lettre_debut, lettre_debut + dataset.shape[1]):
        sheet[chr(j) + "1"] = dataset.columns[j - lettre_debut]
   
    # Écriture des données
    for i in range(dataset.shape[0]):
        for j in range(lettre_debut, lettre_debut + dataset.shape[1]):
            sheet[chr(j) + str(i + 2)].value = dataset.iloc[i, j - lettre_debut]
   
    # Écriture des moyennes et des écarts types
    moyenne = ["moy indice cubital", "moyenne hantel"]
    ecart_type = ["E-T indice cubital", "E-T hantel"]
   
    for i in range(dataset.shape[1]):
        # Ecriture des moyennes
        sheet[chr(lettre_debut + 2 + dataset.shape[1]) + str(i + 1)] = moyenne[i]
        sheet[chr(lettre_debut + 3 + dataset.shape[1]) + str(i + 1)].value = dataset.iloc[:, i].mean()
       
        # Ecriture des écarts types
        sheet[chr(lettre_debut + 4 + dataset.shape[1]) + str(i + 1)] = ecart_type[i]
        sheet[chr(lettre_debut + 5 + dataset.shape[1]) + str(i + 1)].value = dataset.iloc[:, i].std()
   
    # Sauvegarde
    travail.save(file_path)
    print("\nSauvegarde effectuée")
