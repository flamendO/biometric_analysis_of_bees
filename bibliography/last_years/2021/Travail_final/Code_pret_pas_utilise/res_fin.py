#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:48:21 2021

@author: chloepoulic
"""


import os
from tkinter import filedialog, PhotoImage, Canvas
from resultats import histogramme_cubital, toile_araignee, histo_Hantel, class_Dreher
import datetime
import tkinter as tk



# fonction qui permet de modifier le dossier qui a été créé
def mkdir_with_mode(directory, mode):
    if not os.path.isdir(directory):
        oldmask = os.umask(0o0)
        os.makedirs(directory, 0o777)
        os.umask(oldmask)

def creation_dossier(path_ou_creer_dossier):
    mode = 0o666
    now = datetime.datetime.now()
    now_char = now.strftime("%H-%M-%S_%d/%m/%Y")
    new_path = path_ou_creer_dossier + '/'+ now_char
    mkdir_with_mode(new_path, mode)
    return new_path 


def lancement_res(fen, indice_cubital, indice_hantel, nombre, classes, indice_disco):
        #demander ou je le range
        path = filedialog.askdirectory(title="Répertoire de travail")
        new_path = creation_dossier(path)

        #execution des tracés indices
        histogramme_cubital(indice_cubital, new_path)
        toile_araignee(indice_disco, indice_cubital, new_path)
        histo_Hantel(indice_hantel, new_path)
        class_Dreher(nombre, classes, new_path)
        
        #affichage rapide des résultats et commentaires
        #on va chercher les images, puis on les mets dans une fenêtre
        
        #racine.destroy() détruire l'ancienne fenetre 
        res_fen = tk.Tk() 
        res_fen.geometry("1080x720")
        res_fen.title("BeeAPI")
        res_fen.minsize(480, 360)
        res_fen.iconbitmap('icone.ico')
        res_fen.config(background = '#FFCC66') 
        
        
        frame_res= tk.Frame(res_fen, bg = '#FFCC66')
        width = 1080
        height = 720
        
        #import de l'image et positionnement
        image_1 = PhotoImage(file = new_path + "/histo_cub.png")
        image_2 = PhotoImage(file = new_path + "/toile_araignee.png")
        image_3 = PhotoImage(file = new_path + "/histo_hantel.png")
        image_4 = PhotoImage(file = new_path + "/classe_dreher.png")
        
        canvas = Canvas(frame_res, width=width, height = height, bg = "#FFCC66", bd = 0, highlightthickness = 0)
        
        canvas.create_image(width/2, height/2, image = image_1) #a changer
        canvas.create_image(width/2, height/2, image = image_2) #a changer
        canvas.create_image(width/2, height/2, image = image_3) #a changer
        canvas.create_image(width/2, height/2, image = image_4) #a changer
        canvas.pack(expand = tk.YES)
        
        #Création de la liste
        txt =  "Nous avons stocké tous ces résultats à l'adresse de votre ordinateur : " + path
        instruc= tk.Label(fen, text =txt, font = "arial 17 bold", bg = "#FFCC66", wraplength = 400, justify = tk.LEFT)
        instruc.pack(side = tk.BOTTOM)
        frame_res.pack(expand = tk.YES)
