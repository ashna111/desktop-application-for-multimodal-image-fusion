from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import cv2

# Root window
root=Tk()
root.title('Multi-modal medical image fusion to detect brain tumors')
# root.geometry('900x500')

ct=cv2.imread(r'C:\Users\Kushal\Desktop\ct.jpg')
ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY)

ct_registered_image = Image.fromarray(ct)
ct_registered_image = ImageTk.PhotoImage(image=ct_registered_image) 
ct_registered_image_label=Label(image=ct_registered_image)
ct_registered_image_label.grid(row=1,column=1)

root.mainloop()
