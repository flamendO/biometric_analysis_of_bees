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
from tkinter.font import Font
from detection_points_nouveau import detection_point

# customtkinter.set_appearance_mode("dark")
# customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.attributes("-fullscreen", False)  # Mettre en plein écran
root.geometry("1000x600")
root.resizable(False, False)

bg="black"
root.config(bg=bg)

filename = ""
progress_bar = None

def exit():
    root.destroy()

def importer():
    global filename
    #print(filename)

def open_file_dialog():
    global filename
    global label1  # Déclarer label1 comme global pour pouvoir le réutiliser dans cette fonction
    filename = filedialog.askopenfilename()
    if filename:
        # Supprimer l'image précédente si elle existe
        if 'label1' in globals():
            label1.destroy()
        image = customtkinter.CTkImage(light_image=Image.open(filename), dark_image=Image.open(filename), size=(400, 250))
        label1 = customtkinter.CTkLabel(root, text="", image=image)
        label1.pack(expand=True, pady=20, padx=60)
        return filename


def open_bar():
    global progress_bar
    progress_window = tk.Toplevel(root)
    progress_window.title("Generating")
    progress_window.geometry("300x100")
    
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="indeterminate")
    progress_bar.pack(pady=20)
    perform_analysis()
    progress_window.destroy()



def perform_analysis():
    global progress_bar
    global images_list
    images_list=[]

    progress_bar.start()
    
    # Chemin du fichier 
    path = filename

    # Séparation des ailes 
    indice = wing_extraction(path)

    # Rotation des ailes 
    angles = np.zeros((indice)) # Array avec les angles de rotation de chaque image si nécessaire
    os.chdir("./save")
    for i in range(1, indice+1):
        image_tmp, angle_tmp = rotate_wing(os.path.join(os.getcwd(), str(i) + '.png'))
        angles[i-1] = angle_tmp
        images_list.append(np.array(image_tmp))
        #print(os.path.join(os.getcwd(), str(i) + '.png'))
        
        progress_value = (i / indice) * 100 # mise a jour barre de progression
        # progress_bar["value"] = progress_value
        root.update_idletasks()
        
    detection_point(images_list[2])
    os.chdir("../")
    progress_bar.stop()
    
    
    # plt.imshow(images_list[12])
    # plt.show()
    


def results():
    print("A FAIRE")



# frame = customtkinter.CTkFrame(master=root)
# frame.pack(pady=10, padx=10, expand=False)


####### LOGO

img_logo = Image.open("./images/logo.png")
resized_logo = img_logo.resize((100,100), Image.Resampling.LANCZOS)
img_logo_resized = ImageTk.PhotoImage(resized_logo)

logo = tk.Label(root, image=img_logo_resized, bg=bg, fg="white" )

logo.pack(side=tk.TOP,pady=10)

# label = customtkinter.CTkLabel(master=frame, text="Projet Abeille", font=("Roboto", 56))
# label.pack(pady=12, padx=10)
########

font1=customtkinter.CTkFont(family="Sans serif", size=20, weight="bold")

# button_frame = customtkinter.CTkFrame(master=root)  # Créer un cadre pour les boutons
# button_frame.pack(side=tk.BOTTOM, pady=80, padx=100)  # Ajouter une marge en haut

button1 = customtkinter.CTkButton(master=root, text="IMPORTER", command=open_file_dialog, font=font1)
button1.place(x=50, y=150)

button2 = customtkinter.CTkButton(master=root, text="ANALYSE", command=open_bar, font=font1)
button2.place(x=300, y=150)

button3 = customtkinter.CTkButton(master=root, text="RÉSULTATS", command=results, font=font1)
button3.place(x=550, y=150)

button4 = customtkinter.CTkButton(master=root, text="QUITTER", command=exit, font=font1)
button4.place(x=800, y=150)



root.mainloop()

