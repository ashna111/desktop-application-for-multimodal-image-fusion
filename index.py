from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog

# Root window
root=Tk()
root.title('Multi-modal medical image fusion to detect brain tumors')
# root.geometry('900x500')

# Upload Files frame
frame_file=LabelFrame(root, text="Select files:", padx=300,pady=20)
frame_file.pack()

# Display Uploaded Images
frame_images=LabelFrame(root, text="The Selected Files",pady=20)
frame_images.pack()

def openMRI():
    global my_image_1

    root.filename=filedialog.askopenfilename(initialdir="/", title="Select MRI Image")
    my_label_1=Label(frame_images,text="MRI Image").grid(row=1,column=0)
    my_image_1=ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label_1=Label(frame_images,image=my_image_1).grid(row=2,column=0,padx=20,pady=20)

def openCT():
    global my_image_2

    root.filename=filedialog.askopenfilename(initialdir="/", title="Select CT Image")
    my_label_2=Label(frame_images,text="CT Image").grid(row=1,column=1)
    my_image_2=ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label_2=Label(frame_images,image=my_image_2).grid(row=2,column=1,padx=20,pady=20)

mri_button=Button(frame_file,text="Select MRI File",command=openMRI)
mri_button.grid(row=0,column=0,pady=10,padx=10)

ct_button=Button(frame_file,text="Select CT File",command=openCT)
ct_button.grid(row=0,column=1,pady=10,padx=10)





root.mainloop()