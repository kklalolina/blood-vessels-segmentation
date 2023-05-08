from tkinter import Tk, Label, Scale, Button, Text, Checkbutton, messagebox, Entry, StringVar, filedialog

from PIL import ImageTk, Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance
import numpy as np

import skimage as sk
from numpy import bool_


class App:

    def __init__(self):

        
        #display settings
        self.windowWidth = 2500
        self.windowHeight = 1300
        self.maxImageDisplayWidth = 600
        self.maxImageDisplayHeight = 600


        # window
        self.window = Tk()
        self.window.title('blood vessels segmentation')
        self.window.geometry('{width}x{height}'.format(width=self.windowWidth, height=self.windowHeight))

        # images
        self.images = []
        self.imagePIs = []
        self.imageLabels = []


        # computation status

        self.statusLabel = Label(self.window, text="", font='Helvetica 18 bold')

        # file selction
        selectFileTitleLabel = Label(self.window, text="Wybór pliku:", font='Helvetica 18 bold')
        selectFileButton = Button(self.window, text="wybierz obraz wejściowy", command=self.selectFile)
        self.selectedFileNameLabel = Label(self.window, text="nie wybrano pliku")



        startButton = Button(self.window, text="Start", width=10, command=self.detectVessels)


        # layout
        startButton.grid(column=1,row=5)


        self.statusLabel.grid(column=4, row=4)

        selectFileTitleLabel.grid(column=0, row = 4, sticky='W')
        selectFileButton.grid(column=1, row = 4)
        self.selectedFileNameLabel.grid(column=2, row=4)

        self.setImage(Image.open('images/01_dr.JPG'), 0)

        self.window.mainloop()

    def resizeProportionally(self, image, maxWidth, maxheight):
        if image.size[0] > image.size[1]:
            newShape = (maxheight, int(maxheight * (image.size[1] / image.size[0])) + 1)
        else:
            newShape = (int(maxWidth * (image.size[0] / image.size[1])) + 1, maxWidth)

        return image.resize(newShape, resample=Image.BICUBIC)

    # returns image resized to fit in limits set in maxImageDisplayWidth and maxImageDisplayHeight variables
    def resizeToFitLimits(self, image):
        return self.resizeProportionally(image, self.maxImageDisplayWidth, self.maxImageDisplayHeight)

    # wyświetla obrazek na podanej pozycji (lub na nowej, jeżeli nie podano)
    def setImage(self, image, index = -1):
            if index == -1 or index >= len(self.images):
                index = len(self.images)
                self.images.append(image)
                self.imagePIs.append(ImageTk.PhotoImage(self.resizeToFitLimits(self.images[index])))
                self.imageLabels.append(Label(self.window, image=self.imagePIs[index]))
                row = index // (self.windowWidth // self.maxImageDisplayWidth)
                col = index % ((self.windowWidth // self.maxImageDisplayWidth))
                self.imageLabels[index].grid(column=col, row=row)
            else:
                self.images[index] = image
                self.imagePIs[index] = ImageTk.PhotoImage(self.resizeToFitLimits(self.images[index]))
                self.imageLabels[index].configure(image=self.imagePIs[index])
    
    def getImage(self, index):
        if index >= len(self.images):
            return Image.open('blank.png')
        return self.images[index]

    

    # event handlers
    def selectFile(self):
        filename = filedialog.askopenfilename()

        if filename != '':
            self.inputImageFile = filename
            self.selectedFileNameLabel.config(text=filename[filename.rindex('/')+1:])

            self.setImage(Image.open(filename), 0)


    def detectVessels(self):

        # używamy tylko kanału M (tak mi działa lepiej niż jakieś kanały z RGB) 
        resimage = self.getImage(0).convert('CMYK').getchannel('M')

        # ze względu na użyte filtru działa tylko dla małych obrazków, trzeba to będzie potem zmienić
        resimage = self.resizeProportionally(resimage, 600, 600)

        # podbijamy kontrast
        resimage = ImageEnhance.Contrast(resimage).enhance(1.8)

        # znajdujemy krawędzie
        resimage = resimage.filter(ImageFilter.FIND_EDGES)

        # rozmycie
        resimage = resimage.filter(ImageFilter.BoxBlur(1))

        # progowe mapowanie na obraz czarno-biały
        resimage = resimage.point( lambda p: 255 if p > 20 else 0 )

        # wyświetlenie wyniku
        self.setImage(resimage, 1)

        # zapis do pliku
        resimage.save("output.png")