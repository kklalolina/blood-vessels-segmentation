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
        # self.setImage(resimage, 1)

        # zapis do pliku
        resimage.save("output.png")
        self.createErrorMatrix()

    def createErrorMatrix(self):
        # kolory w RGB
        WHITE = (255,255,255)
        BLACK = (0,0,0)
        RED = (255,0,0)
        GREEN = (0,255,0)

        # macierz pomyłek
        errormatrix = np.array([[0,0],[0,0]])

        # wzięcie odpowiedniego obrazu 'idealnego' (z folderu manual1) na podstawie wejściowego
        inputimgname = self.images[0].filename
        inputimgname = inputimgname.replace('images', 'manual1')
        inputimgname = inputimgname[:-4]+'.tif'


        # Gold Standard image
        gsimg = Image.open(inputimgname).convert('RGB')
        gsimg = self.resizeProportionally(gsimg, 600, 600)
        gsimg = gsimg.point(lambda p: 255 if p > 20 else 0)
        gsimg.save("scaled.png")
        # nasz obraz z wykrytymi naczyniami
        outputimg = Image.open('output.png').convert('RGB')

        # porównujemy nasz obraz z gold standard i uzupełniamy macierz pomyłek jednocześnie zaznaczając pomyłki kolorami na obrazie
        for x in range(outputimg.width):
            for y in range(outputimg.height):
                if outputimg.getpixel((x, y)) == WHITE and gsimg.getpixel((x, y)) == BLACK:
                    outputimg.putpixel((x,y),RED)
                    errormatrix[0,1] += 1
                elif outputimg.getpixel((x, y)) == BLACK and gsimg.getpixel((x, y)) == WHITE:
                    outputimg.putpixel((x,y),GREEN)
                    errormatrix[1, 0] += 1
                elif outputimg.getpixel((x, y)) == WHITE and gsimg.getpixel((x, y)) == WHITE:
                    errormatrix[0, 0] += 1
                else:
                    errormatrix[1, 1] += 1

        # wyświetlenie wyniku
        self.setImage(outputimg, 1)

        # wypisanie macierzy pomyłek (na konsole)
        self.printErrorMatrix(errormatrix)

        # wyliczenie miar skuteczności i dla danych niezrównoważonych
        self.getMOEs(errormatrix)

        # zapis do pliku
        outputimg.save("output1.png")

    def printErrorMatrix(self, matrix):
        row_names = ['PP', 'PN']
        column_names = ['AP', 'AN']

        seplen = max(len(str(matrix[0, 0])), len(str(matrix[1, 0])))
        dif = abs(len(str(matrix[0, 0])) - len(str(matrix[1, 0])))

        row0 = ' ' * 5 + column_names[0] + ' ' * (seplen + 4) + column_names[1]
        print(row0)
        for row in range(2):
            if len(str(matrix[row, 0])) == seplen:
                print(row_names[row], ' ', matrix[row, 0], ' ' * seplen, matrix[row, 1])
            else:
                print(row_names[row], ' ', matrix[row, 0], ' ' * (seplen + dif), matrix[row, 1])
        print()

    def getMOEs(self, matrix):
        TP = matrix[0, 0]
        TN = matrix[1, 1]
        FP = matrix[0, 1]
        FN = matrix[1, 0]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)  # Recall
        specificity = TN / (FP + TN)
        precision = TP / (TP + FP)

        # print('Measures of effectiveness:')
        print('Accuracy:', accuracy)
        print('Sensitivity/Recall:', sensitivity)
        print('Specificity:', specificity)
        print('Precision:', precision)
        print()

        G_mean = np.sqrt(sensitivity * specificity)
        F_measure = (2 * precision * sensitivity) / (precision + sensitivity)

        # print('Measures for unbalanced data:')
        print('G-mean:', G_mean)
        print('F-measure:', F_measure)


