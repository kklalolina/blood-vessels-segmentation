from tkinter import Tk, Label, Scale, Button, Text, Checkbutton, messagebox, Entry, StringVar, filedialog

from PIL import ImageTk, Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance

import pickle
import math
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

# pyplot do zapisania macierzy pomyłek jako wykres
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import time


class App:

    def __init__(self):

        
        #display settings
        self.windowWidth = 1920
        self.windowHeight = 1000
        self.maxImageDisplayWidth = 400
        self.maxImageDisplayHeight = 400

        self.numberOfNeighbors = 6
        self.part_size = 5
        # maksymalna wartość jaką może mieć szerokość i wysokość obrazka (aby obliczenia szybciej przebiegały)
        self.maxSize = 600

        self.output_dir = 'output/'
        self.models_dir = 'models/'

        self.trained = False
        self.trainingSetDirectory = 'training_set_1'


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

        trainButton = Button(self.window, text="Trenuj klasyfikator", width=20, command=self.trainClassifier)
        startButton = Button(self.window, text="Start", width=10, command=self.detectVesselsUsingClassifier)

        neighborsLabel = Label(self.window, text='liczba sąsiadów')
        self.neighborsVar = StringVar()
        neighborsInput = Entry(self.window, textvariable=self.neighborsVar, width = 10, justify='center', validate="key")
        self.neighborsVar.set(self.numberOfNeighbors)

        partsizeLabel = Label(self.window, text='rozmiar fragmentu')
        self.partsizeVar = StringVar()
        partsizeInput = Entry(self.window, textvariable=self.partsizeVar, width = 10, justify='center', validate="key")
        self.partsizeVar.set(self.part_size)
        
        selectTrainingSetButton = Button(self.window, text="Wybierz zbiór treningowy", width=20, command=self.selectTrainingSet)

        # saving and loading model controls

        saveModelButton = Button(self.window, text="Zapisz klasyfikator", width=20, command=self.saveModel)
        self.modelFilenameVar = StringVar()
        modelFilenameInput = Entry(self.window, textvariable=self.modelFilenameVar, width = 30)
        self.modelFilenameVar.set('nazwa_nowego_modelu')

        loadModelButton = Button(self.window, text="Wczytaj klasyfikator", width=20, command=self.loadModel)

        self.currentModelLabel = Label(self.window, text='Nie wczytano modelu')

        # layout
        startButton.grid(column=1,row=5)


        neighborsLabel.grid(column=0, row=8, sticky='N')
        neighborsInput.grid(column=0, row=9)

        partsizeLabel.grid(column=0, row=10, sticky='N')
        partsizeInput.grid(column=0, row=11)
        selectTrainingSetButton.grid(column=0, row=12, pady='10')
        trainButton.grid(column=0, row=13, pady='10')

        self.statusLabel.grid(column=4, row=4)

        selectFileTitleLabel.grid(column=0, row = 4, sticky='W')
        selectFileButton.grid(column=1, row = 4)
        self.selectedFileNameLabel.grid(column=2, row=4)

        saveModelButton.grid(column=2, row=9)
        modelFilenameInput.grid(column=3, row=9)
        loadModelButton.grid(column=2, row=10)
        self.currentModelLabel.grid(column=2, row=11)

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

    # Folder ze zbiorem treningowym musi zwierać dwa fodlery: images i manual. W folderze images znajdują się zdjęcia siatkówki, a w manual odpowiadające im maski eksperckie. Po posortowaniu plików po nazwach odpowiadjące sobie obrazy muszą być na tych samych pozycjach. 
    def selectTrainingSet(self):
        directory = filedialog.askdirectory()

        if directory != '':
            self.trainingSetDirectory = directory 

    def saveModel(self):

        if not self.trained:
            self.statusLabel.config(text='Brak modelu do zapisania')
            return
        
        filename = self.modelFilenameVar.get()
        filename += '.knn'

        if filename != '':
            file = open(self.models_dir + filename, 'wb')
            pickle.dump(self.knn, file)
            file.close()
            self.currentModelLabel.config(text=filename)
            self.statusLabel.config(text='zapisano model')

    def loadModel(self):
        
        filename = filedialog.askopenfilename()

        if filename != '':
            file = open(filename, 'rb')
            self.knn = pickle.load(file)
            file.close
            self.trained = True
            self.currentModelLabel.config(text=filename[filename.rindex('/')+1:])
            self.statusLabel.config(text='wczytano model')

    def applyParams(self):
        self.part_size = int(self.partsizeVar.get())
        self.numberOfNeighbors = int(self.neighborsVar.get())

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
        self.setImage(resimage, 3)

        # zapis do pliku
        resimage.save(self.output_dir + "output1.png")
        # self.createErrorMatrix()

    def preprocessImage(self, image):

        # zmiejszamy obrazek żeby obliczenia szybciej przebiegły
        image = self.resizeProportionally(image, self.maxSize, self.maxSize)

        # używamy tylko kanału M (tak mi działa lepiej niż jakieś kanały z RGB) 
        image = image.convert('CMYK').getchannel('M')

        # podbijamy kontrast
        image = ImageEnhance.Contrast(image).enhance(1.8)

        return image
    
    def preprocessManual(self, manual):

        manual = manual.convert('L')

        # zmiejszamy obrazek żeby obliczenia szybciej przebiegły
        manual = self.resizeProportionally(manual, self.maxSize, self.maxSize)

        manual = manual.point(lambda p: 255 if p > 20 else 0)
        manual.save(self.output_dir + "scaled.png")

        return manual

    def postprocess(self, image):
        WHITE = np.array([255, 255, 255])
        BLACK = np.array([0, 0, 0])

        # usuwanie pojedynczych bialych pikseli z tla
        resimage = image.copy()
        d = 1
        tr = 2 * 765
        for x in range(d, image.shape[0]-d):
            for y in range(d, image.shape[1]-d):
                sum = np.sum(image[x-d:x+d+1, y-d:y+d+1])
                if sum <= tr:
                    resimage[x, y] = BLACK

        # uzupelnianie ubytkow
        d = 1
        tc = ((2*d+1)*(2*d+1) - 4) * 765
        for x in range(d, image.shape[0]-d):
            for y in range(d, image.shape[1]-d):
                sum = np.sum(image[x-d:x+d+1, y-d:y+d+1])
                if sum >= tc:
                    resimage[x, y] = WHITE

        return resimage


    def getManual(self, inputimgname):
        # wzięcie odpowiedniego obrazu 'idealnego' (z folderu manual1) na podstawie wejściowego
        inputimgname = inputimgname.replace('images', 'manual1')
        inputimgname = inputimgname[:-4] + '.tif'

        # Gold Standard image
        gsimg = Image.open(inputimgname).convert('RGB')
        gsimg = self.resizeProportionally(gsimg, self.maxSize, self.maxSize)
        gsimg = gsimg.point(lambda p: 255 if p > 20 else 0)
        gsimg.save(self.output_dir + "scaled.png")
        return gsimg

    # from https://learnopencv.com/shape-matching-using-hu-moments-c-python/
    def get_hu_moments(self, img_part):
        # Calculate Moments
        moments = cv2.moments(img_part)
        # Calculate Hu Moments
        huMoments = cv2.HuMoments(moments)

        # tu zmieniłem na operacje na tablicach za względu na wydajność
        # Log scale hu moments (After the following transformation, the moments are of comparable scale)
        huMoments = -1 * np.sign(huMoments) * np.log10(np.abs(huMoments),  where=(huMoments != 0))
        # for i, _ in enumerate(huMoments):
        #     if huMoments[i] != 0:
        #         huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))


        return [moment[0] for moment in huMoments]


    def trainClassifier(self):

        start_time = time.time()

        self.applyParams()

        # inicjujemy klasyfikator knn podając liczbę sąsiadów
        self.knn = KNeighborsClassifier(n_neighbors=self.numberOfNeighbors)


        imagesDirectory = self.trainingSetDirectory + '/images'
        manualDirectory = self.trainingSetDirectory + '/manual'
        images = [f for f in listdir(imagesDirectory) if isfile(join(imagesDirectory, f))]
        images.sort()
        manuals = [f for f in listdir(manualDirectory) if isfile(join(manualDirectory, f))]
        manuals.sort()

        for fid in range(len(images)):

            # bierzemy obrazek i na jego podstawie klasyfikujemy
            image = Image.open(join(imagesDirectory, images[fid]))

            # bierzemy odpowiadajaca maske ekspercka
            manual = Image.open(join(manualDirectory, manuals[fid]))

            image = self.preprocessImage(image)
            manual = self.preprocessManual(manual)

            image = np.array(image)
            manual = np.array(manual)
            manual = manual.flatten()

            height, width = image.shape

            # rozmiar fragmentow z obrazka dla ktorych bedziemy obliczać Hu momenty
            part_size = self.part_size

            parts = []
            # iterujemy po każdym pikselu w obrazku, dany piksel jest środkiem naszego fragmentu o rozmiarze part_size x part_size
            for y in range(height):
                for x in range(width):
                    center_y = y - part_size // 2
                    center_x = x - part_size // 2

                    if center_y >= 0 and center_y + part_size <= height and center_x >= 0 and center_x + part_size <= width:
                        part = image[center_y:center_y + part_size, center_x:center_x + part_size]
                        parts.append(part)
                    else:
                        # jeżeli piksel brzegowy to bierzemy piksele z fragmentu mieszczace sie w obrazku
                        start_y = max(0, center_y)
                        end_y = min(height, center_y + part_size)
                        start_x = max(0, center_x)
                        end_x = min(width, center_x + part_size)
                        part = image[start_y:end_y, start_x:end_x]
                        parts.append(part)

            huMoments = []

            # obliczamy hu moment dla każdego fragmentu
            for part in parts:
                huMoments.append(self.get_hu_moments(part))

            # trenujemy klasyfikator podając hu momenty i odpowiadające wartości pikseli z maski eksperckiej
            self.knn.fit(huMoments, manual)

        self.trained = True
        self.currentModelLabel.config(text='Nowy Model')

        end_time = time.time()
        print("training: ", end_time-start_time, "s")

    def detectVesselsUsingClassifier(self):

        # jeżeli nie wytrenowaliśmy ani nie wczytaliśmy klasyfikatora to trenujemy teraz
        if not self.trained:
            self.trainClassifier()

        # bierzemy obrazek wejściowy i zmniejszamy aby obliczenia przebiegły szybciej
        image = self.getImage(0)
        image = self.preprocessImage(image)
        image = np.array(image)
        

        height, width = image.shape

        # rozmiar fragmentow z obrazka dla ktorych bedziemy obliczać Hu momenty
        part_size = self.part_size

        result = np.zeros((height, width, 3), dtype=np.uint8)

        start_time = time.time()

        hu_moments = []
        # iterujemy po każdym pikselu w obrazku, dany piksel jest środkiem naszego fragmentu o rozmiarze part_size x part_size
        for y in range(height):
            for x in range(width):
                center_y = y - part_size // 2
                center_x = x - part_size // 2

                if center_y >= 0 and center_y + part_size <= height and center_x >= 0 and center_x + part_size <= width:
                    part = image[center_y:center_y + part_size, center_x:center_x + part_size]
                else:
                    # jeżeli piksel brzegowy to bierzemy piksele z fragmentu mieszczace sie w obrazku
                    start_y = max(0, center_y)
                    end_y = min(height, center_y + part_size)
                    start_x = max(0, center_x)
                    end_x = min(width, center_x + part_size)
                    part = image[start_y:end_y, start_x:end_x]
                # obliczamy hu moment fragmentu i wrzucamy go do tablicy
                hu_moments.append(self.get_hu_moments(part))

        end_time = time.time()
        print("hu_moments: ", end_time-start_time, "s")

        start_time = time.time()

        WHITE = np.array([255, 255, 255])
        # zrobienie predykcji raz na całej tablicy jest znacznie szybsze niż pojedyńczo na każdym elemencie
        # klasyfikator zwraca [0] jeżeli nie naczynie i [255] jeżeli naczynie
        # jeżeli wykryło naczynie to zmieniamy piksel obrazku wynikowym na biały
        pred = self.knn.predict(hu_moments)
        i = 0
        for y in range(height):
            for x in range(width):
                if pred[i] == 255:
                    result[y, x] = WHITE
                i += 1


        result = self.postprocess(result)
        # Image.fromarray i zapisywanie PILem nie działało, bo wymaga uint8 jako typu danych w tablicy, już działa
        result = Image.fromarray(result)
        result.save(self.output_dir + 'output.png')

        self.setImage(result, 1)
        self.createErrorMatrix(result)

        end_time = time.time()
        print("prediction: ", end_time-start_time, "s")



    def createErrorMatrix(self, image):
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
        gsimg = gsimg.resize(image.size)
        gsimg = gsimg.point(lambda p: 255 if p > 20 else 0)
        gsimg.save(self.output_dir + "scaled.png")
        # nasz obraz z wykrytymi naczyniami
        outputimg = Image.open(self.output_dir + 'output.png').convert('RGB')

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
        self.setImage(outputimg, 2)

        # wypisanie macierzy pomyłek (na konsole)
        self.printErrorMatrix(errormatrix)


        # wyliczenie miar skuteczności i dla danych niezrównoważonych
        self.getMOEs(errormatrix)

        # zapis do pliku
        outputimg.save(self.output_dir + "output1.png")

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

        # zapisanie i wyświetlenie macierzy jako wykres
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', 'green'])

        plt.imshow(matrix, cmap=custom_cmap)

        x_ticks = np.arange(len(matrix[0]))
        y_ticks = np.arange(len(matrix))

        x_labels = column_names
        y_labels = row_names

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                plt.text(j, i, str(matrix[i][j]), ha='center', va='center', color='black')

        plt.xticks(x_ticks, x_labels)
        plt.yticks(y_ticks, y_labels)

        plt.colorbar()
        plt.savefig(self.output_dir + 'matrix.png')

        matrixImg = Image.open(self.output_dir + 'matrix.png')
        self.setImage(matrixImg, 3)


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


