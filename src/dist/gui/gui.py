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
from excel import toExcel
import pandas as pd
import sys
import shutil
import time
import ctypes
import webbrowser

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
affichage_result_key = False


def exit(): # Termine le programme proprement 
    global root
    root.quit()  
    root.destroy()  
    sys.exit()  

def importer():
    global filename

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
    if not filename:
        messagebox.showerror("Erreur", "Aucune image n'a été selectionné !")
    else:
        
        progress_window = tk.Toplevel(root)
        progress_window.title("Generating")
        progress_window.geometry("300x100")
    
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="indeterminate")
        progress_bar.pack(pady=20)
        perform_analysis()
        progress_window.destroy()



def perform_analysis():
    tmp_path = "./tmp/"
    # shutil.rmtree(tmp_path)
    # os.mkdir("./tmp/")

    for file_name in os.listdir(tmp_path):
        file_path = os.path.join(tmp_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erreur lors de la suppression de {file_path}: {e}")

    global filename
    
    global progress_bar
    global images_list
    images_list=[]

    progress_bar.start()
    
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
        
        root.update_idletasks()


    global data_set
    global df
    data_set = np.zeros((14,2)) # 2 colonnes qui correspondent aux indices cubitales et anthem
    i = 0
    for img_retourne in images_list:
        indice_cubi , indice_anthem = detection_point(img_retourne)
        
        data_set[i,:] = np.array([indice_cubi, indice_anthem])
        i+=1
    progress_bar.stop()
    os.chdir("../")
    columns = ['Indice cubitale', 'Indice Hantel']
    df = pd.DataFrame(data_set, columns=columns)
    

    # font2=customtkinter.CTkFont(family="Sans serif", size=20, weight="bold")
    # affichage_result = tk.Label(root, text="ANALYSE TERMINÉE ! ", font=font2, bg=bg, fg="white")
    # affichage_result.place(y=500, x=400)

    messagebox.showinfo("Terminé", "ANALYSE TERMINÉE !")



    
    



def results():
    font2=customtkinter.CTkFont(family="Sans serif", size=20, weight="bold")
    global df

    if 'df' not in globals() or df.empty:
        messagebox.showerror("Erreur", "Aucune analyse n'a été effectuée.")
    else:
        if messagebox.askyesno("Confirmation", "Souhaitez-vous sauvegarder les images de l'analyse ?"):
            # Ouvrir le sélecteur de dossier pour sélectionner le dossier de sauvegarde des images
            dossier_images = filedialog.askdirectory()
            if dossier_images:
                
                shutil.move("./tmp/", dossier_images)
                os.mkdir("./tmp")

        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Fichiers Excel", "*.xlsx"), ("Tous les fichiers", "*.*")])
        
        if filepath:
            toExcel(df, filepath)
            # affichage_result_2 = tk.Label(root, text="FICHIER SAUVEGARDÉ ! ",font=font2, bg=bg, fg="white" )
            # affichage_result_2.place(x= 395, y = 550)
            messagebox.showinfo("Terminé", "FICHIER SAUVEGARDÉ ! ")
        


def help():
    url = "https://github.com/flamendO/biometric_analysis_of_bees"
    webbrowser.open(url)

####### LOGO

img_logo = Image.open("./images_logo/logo.png")
resized_logo = img_logo.resize((100,100), Image.Resampling.LANCZOS)
img_logo_resized = ImageTk.PhotoImage(resized_logo)

logo = tk.Label(root, image=img_logo_resized, bg=bg, fg="white" )

logo.pack(side=tk.TOP,pady=10)



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

button5 = customtkinter.CTkButton(master=root, text="AIDE / DOCUMENTATION", command=help, font=font1)
button5.place(x=380, y=500)


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)

if is_admin():
    root.mainloop()
else:
    run_as_admin()
    root.mainloop()






