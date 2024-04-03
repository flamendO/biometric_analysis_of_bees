import customtkinter
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from main import main
from tkinter import messagebox
from tkinter import ttk
import threading
import numpy as np
import matplotlib.pyplot as plt
import os
from wing_extraction import wing_extraction
from rotation_aile import rotate_wing

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("1000x600")


filename = ""
progress_bar = None

def importer():
    global filename
    #print(filename)

def open_file_dialog():
    global filename
    filename = filedialog.askopenfilename()
    if filename:
        image=customtkinter.CTkImage(light_image=Image.open(filename), dark_image=Image.open(filename), size=(400,250))
        label1 = customtkinter.CTkLabel(root, text="", image=image)
        label1.place(x=100,y=150)
        return filename
    
def analyse():
    global progress_bar
    if filename == "":
        tk.messagebox.showerror("Erreur", message="Aucun fichier sélectionné")
    else:
        progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        progress_bar.place(x=300, y=450)
        progress_bar["value"] = 0  # Initialiser la valeur de la barre à 0

        # Exécuter le processus d'analyse dans un thread séparé
        analysis_thread = threading.Thread(target=perform_analysis)
        analysis_thread.start()

def perform_analysis():
    global progress_bar
    global images_list
    images_list=[]
    
    # Chemin du fichier 
    path = filename

    # Séparation des ailes 
    indice = wing_extraction(path)

    # Rotation des ailes 
    angles = np.zeros((indice)) # Array avec les angles de rotation de chaque image si nécessaire
    
    for i in range(1, indice+1):
        image_tmp, angle_tmp = rotate_wing(os.path.join(os.getcwd(), str(i) + '.png'))
        angles[i-1] = angle_tmp
        images_list.append(np.array(image_tmp))
        #print(os.path.join(os.getcwd(), str(i) + '.png'))
        
        
        progress_value = (i / indice) * 100 # mise a jour barre de progression
        progress_bar["value"] = progress_value
        root.update_idletasks()

    # plt.imshow(images_list[12])
    # plt.show()



frame = customtkinter.CTkFrame(master=root)
frame.grid(row=0, column=0, stick='news')
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Projet Abeille", font=("Roboto",56))
label.pack(pady=12, padx=10)








button1 = customtkinter.CTkButton(master=frame, text="Importer", command=open_file_dialog)

button1.pack(pady=300, padx=0)
button1.place(x=100,y=500)

# button2 = customtkinter.CTkButton(master=frame, text="Analyse", command=analyse)
# button2.pack(expand=True)

# button1.grid(row=2, column=0)


## FRAME 1

frame1 = customtkinter.CTkFrame(master=root, fg_color="#8D6F3A", border_color="#FFCC70", border_width=2)

frame.pack(expand=True)


root.mainloop()





