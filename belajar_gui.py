import tkinter as tk
from tkinter import ttk #tema/tampilan
from tkinter.messagebox import showinfo
window = tk.Tk()
window.configure(bg="white")
window.geometry("300x200")
window.resizable(False, False) #sumbu x y window tidak bisa diubah ukurannya
window.title("Sapa Dia")

#frame input
input_frame = ttk.Frame(window) #wadah untuk mengelompokkan komponen GUI
#penempatan ada 3 : grid,pack,place
input_frame.pack(padx=10,pady=10,fill="x",expand=True)
#padx=jarak kiri kanan, pady=jarak atas bawah, fill= melebar horizontal(x)/ vertikal(y), expand=menyesuaikan ruang

#variabel dan fungsi
NAMA_DEPAN = tk.StringVar() #menggunakan stringvar agar jika nama berubah maka akan lebih mudah mengolahnya dan menyimpan teks dari entry
NAMA_BELAKANG = tk.StringVar()

def tombol_click():
   #fungsi ini akandipanggil oleh tombol
    pesan = f"1 es teh {NAMA_DEPAN.get()} {NAMA_BELAKANG.get()}, cakepp!"
    showinfo(title="uy!",message=pesan)
    #pop up seperti alert dalam js

#komponen-komponen
#1. label nama depan
nama_depan_label = ttk.Label(input_frame,text="Nama Depan:")
nama_depan_label.pack(padx=10,fill="x",expand=True)

#2. entry nama depan (Entry adalah widget (komponen GUI) yang digunakan untuk membuat kotak input teks agar pengguna bisa mengetik data di aplikasi.)
nama_depan_entry = ttk.Entry(input_frame,textvariable=NAMA_DEPAN)
nama_depan_entry.pack(padx=10,fill="x",expand=True)

#3. label nama belakang
nama_belakang_label = ttk.Label(input_frame,text="Nama Belakang:")
nama_belakang_label.pack(padx=10,fill="x",expand=True)

#4. entry nama belakang
nama_belakang_entry = ttk.Entry(input_frame,textvariable=NAMA_BELAKANG)
nama_belakang_entry.pack(padx=10,fill="x",expand=True)

#5. tombol
tombol_sapa = ttk.Button(input_frame,text="Sapa!!", command=tombol_click)
tombol_sapa.pack(padx=10,pady=10,fill="x",expand=True)

window.mainloop()