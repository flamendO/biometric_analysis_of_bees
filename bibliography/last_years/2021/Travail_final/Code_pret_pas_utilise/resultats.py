#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:32:40 2021

@author: chloepoulic
"""

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from statistics import mean

def histogramme_cubital(indice_cubital, path):
    plt.hist (indice_cubital)
    plt.title("Indice cubital")
    plt.xlabel("Nombre d'échantillon")
    plt.ylabel("Indice cubital")
    plt.grid()
    #plt.show()
    plt.savefig(path + "/histo_cub.png", dpi = 200 )
    #il faut espacer les barres        


def toile_araignee(indice_disco, indice_cubital, path):
    plt.figure(2, figsize=(15,10))
    plt.scatter(indice_disco, indice_cubital, marker = 'o', lw = 4)
    plt.axvline(x=0, color='red', lw = 4)
    plt.axhline(y=2.10, color='red', lw = 4)
    plt.title("Transgression Discoïdale", fontsize=45)
    plt.xlabel("Indice Discoïdal", fontsize=25)
    plt.ylabel("Indice Cubital", fontsize=25)
    plt.grid()
    plt.savefig(path + "/toile_araignee.png", dpi = 200 )
    #plt.show()
    
def histo_Hantel(indice_hantel, path):
    plt.hist(indice_hantel)
    plt.xlim(0.60,1.20)
    plt.xlabel("Nombre")
    plt.ylabel("Indice de Hantel")
    plt.title("Histogramme de Hantel")
    plt.savefig(path + "/histo_hantel.png", dpi = 200 )
    #plt.show()
    
def class_Dreher(nombre, classes, path):
    #défintion des valeurs afin de calculer l'approximation gaussienne
    #x_min = min(classes)
    #x_max = max(classes)
    #moy = mean(classes)
    #var = np.var(classes)
    x_min = min(nombre)
    x_max = max(nombre)
    moy = mean(nombre)
    var = np.var(nombre)
    std = var**2
    x = np.linspace(x_min, x_max, 300)
    y = scipy.stats.norm.pdf(x,moy,std)
    
    ##définition des tracés
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    #Classes
    ax1.bar(classes, nombre, label = "Résultats expérimentaux")
    
    #Gaussienne
    ax2.plot(x,y, color='coral', label = 'Gaussienne associée')
    
    plt.xlabel("Classes")
    plt.ylabel("Nombre")
    plt.title("Répartitions dans les classes Dreher")
    
    
    plt.legend(loc = 'best')
    plt.grid()
    plt.savefig(path + "/classe_dreher.png", dpi = 200 )
    #plt.show()
