import customtkinter
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from main import main
from tkinter import messagebox

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("1000x600")

filename = ""

def importer():
    global filename
    print(filename)

def open_file_dialog():
    global filename
    filename = filedialog.askopenfilename()
    if filename:
        image=customtkinter.CTkImage(light_image=Image.open(filename), dark_image=Image.open(filename), size=(400,250))
        label1 = customtkinter.CTkLabel(root, text="", image=image)
        label1.place(x=100,y=100)
    
        return filename
    
def analyse():
    if filename == "":
        messagebox.showerror("Erreur", message="Auncun fichier sélectionné")
    main(filename)

def loading():
    # BARRE DE CHARGEMENT
    return "ZIZI"



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

button2 = customtkinter.CTkButton(master=frame, text="Analyse", command=analyse)


button2.place(x=400,y=500)
# button1.grid(row=2, column=0)



root.mainloop()





