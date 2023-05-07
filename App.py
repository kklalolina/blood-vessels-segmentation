from tkinter import Tk, Label, Scale, Button, Text, Checkbutton, messagebox, Entry, StringVar, filedialog

from PIL import ImageTk, Image, ImageDraw, ImageOps
import numpy as np

import skimage as sk
from numpy import bool_


class App:

    def __init__(self):


        #display settings
        self.maxImageDisplayWidth = 300
        self.maxImageDisplayHeight = 400


        # window
        self.window = Tk()
        self.window.title('blood vessels segmentation')
        self.window.geometry('1200x500')

        # place for original image 
        self.baseImage = Image.open('images/01_dr.JPG')
        self.img = ImageTk.PhotoImage(self.resizeToFitLimits(self.baseImage))
        self.imgLabel = Label(self.window, image=self.img)
        self.imgLabel.grid(column=0, row=0, columnspan=2)

        # place for filtered image
        self.filtered = Image.open('blank.png')
        self.filImg = ImageTk.PhotoImage(self.resizeToFitLimits(self.filtered))
        self.filImgLabel = Label(self.window, image=self.filImg)
        self.filImgLabel.grid(column=2, row=0, columnspan=2)

        # place for
        self.compareImage = Image.open('blank.png')
        self.comImg = ImageTk.PhotoImage(self.resizeToFitLimits(self.compareImage))
        self.comImgLabel = Label(self.window, image=self.comImg)
        self.comImgLabel.grid(column=4, row=0, columnspan=2)


        # computation status

        self.statusLabel = Label(self.window, text="", font='Helvetica 18 bold')

        # file selction
        selectFileTitleLabel = Label(self.window, text="Wybór pliku:", font='Helvetica 18 bold')
        selectFileButton = Button(self.window, text="wybierz obraz wejściowy", command=self.selectFile)
        self.selectedFileNameLabel = Label(self.window, text="nie wybrano pliku")



        startButton = Button(self.window, text="Start", width=10, command=self.detectVessels)


        # layout
        startButton.grid(column=3,row=5)


        self.statusLabel.grid(column=4, row=2)

        selectFileTitleLabel.grid(column=0, row = 1, sticky='W')
        selectFileButton.grid(column=1, row = 2)
        self.selectedFileNameLabel.grid(column=2, row=2)


        self.window.mainloop()

    # returns image resized to fit in limits set in maxImageDisplayWidth and maxImageDisplayHeight variables
    def resizeToFitLimits(self, image):
        if image.size[0] > image.size[1]:
            newShape = (self.maxImageDisplayHeight, int(self.maxImageDisplayHeight * (image.size[1] / image.size[0])) + 1)
        else:
            newShape = (int(self.maxImageDisplayWidth * (image.size[0] / image.size[1])) + 1, self.maxImageDisplayWidth)

        return image.resize(newShape, resample=Image.BICUBIC)

    # display original image
    def setImage(self, image):
        self.img = ImageTk.PhotoImage(self.resizeToFitLimits(image))
        self.imgLabel.config(image=self.img)

    # display filtered image
    def setFilteredImage(self, image):
        self.filImg = ImageTk.PhotoImage(self.resizeToFitLimits(image))
        self.filImgLabel.config(image=self.filImg)

    

    # event handlers
    def selectFile(self):
        filename = filedialog.askopenfilename()

        if filename != '':
            self.inputImageFile = filename
            self.selectedFileNameLabel.config(text=filename[filename.rindex('/')+1:])


            self.baseImage = Image.open(filename)
            self.setImage(self.baseImage)



    # funkcja w której filtrujemy obrazek przed wyznaczaniem naczyń
    def preprocess(self, image):

        # usuwanie kanału czerwonego (bo tylko takie mamy sprawdzac obrazki?)
        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i][j][0] = 0
        image = Image.fromarray(image, mode='RGB')

        # obrazek w skali szarości
        image = np.array(image.convert('L'))

        # wyostrzenie filtrem Gaussa - chyba trzeba cos lepszego
        blurred = sk.filters.gaussian(image, sigma=2)
        image = image - blurred

        #jakis kontrast todo

        return image

    def findEdges(self, image):
        filtered = sk.filters.hessian(image) # nie musi byc koniecznie ten filtr
        return filtered

    # funckja w której filtrujemy obrazek po znalezieniu naczyń
    def postprocess(self, image):

        # znajdowanie optymalnego progu(do rozrozniania tła) czy to trzeba samemu implementowac?
        threshold = sk.filters.threshold_otsu(image)

        # zamiana image na tablice boolów (True jezeli wartosc > threshold)
        image = image > threshold

        # usunięcie małych obszarów
        image = sk.morphology.remove_small_objects(image, min_size=100)


        # dylatacja o promieniu 2
        image = sk.morphology.dilation(image, sk.morphology.disk(2))

        # nadal zwracamy tablice z bool
        return image


    def detectVessels(self):
        image = np.array(self.baseImage)

        image = self.preprocess(image)
        image = self.findEdges(image)
        image = self.postprocess(image)

        # jak sie zapisuje matplotlibem to czasem obraz inaczej wyglada ;-;
        #resimage = self.saveResult(image)

        resimage = Image.fromarray(image)

        self.setFilteredImage(resimage)

    # do testow, funckja zapisujaca przefiltrowany obraz uzywajac matplotlib
    def saveResult(self, image):
        import matplotlib.pyplot as plt
        # convert to grayscale using matplotlib
        plt.imshow(image.squeeze(), cmap='gray')
        plt.axis('off')
        plt.savefig('output.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        return Image.open('output.png')