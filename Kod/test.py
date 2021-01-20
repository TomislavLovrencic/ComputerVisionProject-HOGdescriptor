import os
from tkinter.ttk import Progressbar

import cv2
import numpy as np
import math
import threading
import time

def preprocessImage(img, negative):
    (height, width, channels) = img.shape
    brightness = 50
    if not negative:
        processedImage = img[(height - 128) // 2:(height + 128) // 2, (width - 64) // 2: (width + 64) // 2]
        # processedImage = img[(width - 64) // 2: (width + 64) // 2, (height - 128) // 2:(height + 128) // 2]
    else:
        processedImage = img[0: height - 1, (width - height // 2) // 2: width - (width - height // 2) // 2]

    # processed_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    processedImage = cv2.resize(processedImage, (64, 128))
    hsv = cv2.cvtColor(processedImage, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - brightness
    v[v > lim] = 255
    v[v <= lim] += brightness

    fhsv = cv2.merge((h, s, v))
    processedImage = cv2.cvtColor(fhsv, cv2.COLOR_HSV2BGR)
    return processedImage


def calculateGradients(img):
    (height, width) = img.shape
    imageGradientH = []
    imageGradientV = []
    imgArr = np.array(img, dtype="int32").tolist()

    # bgrPlane = cv2.MatVector()
    # cv2.split(img, bgrPlane)
    # racunanje horizontalnog gradijenata

    imageGradientH.append([0 for _ in range(0, width)])
    imageGradientV.append([0 for _ in range(0, width)])
    for h in range(1, height - 1):
        hor = [0 for _ in range(0, width)]
        ver = [0 for _ in range(0, width)]
        for w in range(1, width - 1):
            hor[w] = (imgArr[h][w + 1] - imgArr[h][w - 1])
            ver[w] = (imgArr[h + 1][w] - imgArr[h - 1][w])

        imageGradientH.append(hor)
        imageGradientV.append(ver)
    imageGradientH.append([0 for _ in range(0, width)])
    imageGradientV.append([0 for _ in range(0, width)])
    return imageGradientH, imageGradientV


# Testiranje
def hogBGR(img):
    if img.ndim != 3:
        return calculateGradients(img)
    imgGradH_ch = np.empty_like(img, dtype=np.double)
    imgGradV_ch = np.empty_like(img, dtype=np.double)
    imgMagn_ch = np.empty_like(img, dtype=np.double)
    for ch in range(img.shape[2]):
        imgGradH_ch[:, :, ch], imgGradV_ch[:, :, ch] = calculateGradients(img[:, :, ch])
        imgMagn_ch[:, :, ch] = np.hypot(imgGradH_ch[:, :, ch], imgGradV_ch[:, :, ch])
    maxGrad = imgMagn_ch.argmax(axis=2)
    rr, cc = np.meshgrid(np.arange(img.shape[0]),
                         np.arange(img.shape[1]),
                         indexing='ij',
                         sparse=True)
    return imgGradH_ch[rr, cc, maxGrad], imgGradV_ch[rr, cc, maxGrad]


def calcOrientation(gradHor: list, graVert: list):
    directionR = np.arctan2(gradHor, graVert, dtype="float32")
    directionR = np.abs(np.degrees(directionR))
    return directionR


def calculateGradientMagnitudeAndDirection(gradientsHorizontal: np.ndarray, gradientsVertical: np.ndarray):
    magnitudeC = np.hypot(gradientsHorizontal, gradientsVertical)
    directionC = np.arctan(gradientsVertical / (gradientsHorizontal + [1e-8]))
    directionC = np.abs(np.degrees(directionC))
    # print(direction)
    return magnitudeC, directionC


def calculateFeatureVectors(magnitudeF: np.ndarray, directionF: np.ndarray):
    # (height, width) = magnitudeAndDirection.shape
    (height, width) = (128, 64)
    vector = []
    magnitudeF = magnitudeF.tolist()
    directionF = directionF.tolist()
    for y in range(0, height - 7, 8):
        for x in range(0, width - 7, 8):
            histogram = [0 for i in range(0, 9)]
            for i in range(0, 8):
                for j in range(0, 8):
                    if y != 0 and x != 0 and y + i != 127 and x + j != 63:
                        (magnitudeEl, directionEl) = (magnitudeF[y + i][x + j], directionF[y + i][x + j])
                        index = math.floor(directionEl) // 20
                        firstWeight = directionEl - index * 20
                        secondWeight = (index + 1) * 20 - directionEl
                        histogram[index % 9] += firstWeight / 20 * magnitudeEl
                        histogram[(index + 1) % 9] += secondWeight / 20 * magnitudeEl
            vector.append(histogram)

    return vector


def concatenateFeatureVectors(featureVector):
    vector = []
    for y in range(0, 15):
        for x in range(0, 7):
            vec = []
            vec += featureVector[y * 7 + x]
            vec += featureVector[y * 7 + (x + 1)]
            vec += featureVector[(y + 1) * 7 + x]
            vec += featureVector[(y + 1) * 7 + (x + 1)]
            vector.append(vec)
    vector = normalizeVectors(vector)
    return vector


def normalizeVectors(vector):
    for vec in vector:
        norm = np.linalg.norm(vec)
        for i in range(0, len(vec)):
            if norm != 0:
                vec[i] = vec[i] / norm
            else:
                vec[i] = 0
    return vector


def getProjectHOG(img, flag):
    img = preprocessImage(img, flag)
    imageHG, imageVG = hogBGR(img)
    magnitude, direction = calculateGradientMagnitudeAndDirection(imageHG, imageVG)
    fvector = calculateFeatureVectors(magnitude, direction)
    return concatenateFeatureVectors(fvector)


from tkinter import *


class HOGdescriptor(Frame):
    def __init__(self, prozor):
        self.prozor = prozor
        self.prozor.title("*****Hog descriptor GUI*****")
        super().__init__(self.prozor)
        self.grid(rows=7, columns=5, padx=10, pady=10)
        self.Sucelje()
        global broj
        broj = 0
        global broj1
        broj1 = 0
        global broj2
        broj2 = 0

        return

    def Sucelje(self):
        f = ("bold", "14", "calibri")

        self.text1 = StringVar()
        self.text1.set("nestoDrugo")

        self.posFolderTrain = Label(self, text="Positive images folder : ")
        self.posFolderTrain.grid(row=1, column=1, pady=10)

        self.posPathFileTRAIN = StringVar()
        self.posPathFileTRAIN.set("INRIAPerson/Train/pos.lst")
        self.EntryPosTrain = Entry(self, textvariable=self.posPathFileTRAIN)
        self.EntryPosTrain.grid(row=1, column=2, pady=10)

        self.negFolderTrain = Label(self, text="Negative images folder : ")
        self.negFolderTrain.grid(row=2, column=1, pady=10)

        self.negPathFileTRAIN = StringVar()
        self.negPathFileTRAIN.set("INRIAPerson/Train/neg.lst")
        self.EntryNegTrain = Entry(self, textvariable=self.negPathFileTRAIN)
        self.EntryNegTrain.grid(row=2, column=2, pady=10)

        self.TrainOurHog = Button(self, text="Train our HOG", bg="green", fg="white", height=1, width=20,
                                  command=lambda: self.createThread("trainOur"))
        self.TrainOurHog.grid(row=3, column=1)

        self.TrainOpenCvHog = Button(self, text="Train OpenCV HOG", bg="green", fg="white", height=1, width=20,
                                     command=lambda: self.createThread("trainOpenCv"))
        self.TrainOpenCvHog.grid(row=3, column=2)

        self.DefaultLocation = Label(self, text="Use Trained model : ")
        self.DefaultLocation.grid(row=4, column=1, pady=10)

        self.textvariable = StringVar()
        self.textvariable.set("svm")
        self.DefaultLocationInput = Entry(self, textvariable=self.textvariable)
        self.DefaultLocationInput.grid(row=4, column=2, pady=10)

        self.posFolderTest = Label(self, text="Positive images folder : ")
        self.posFolderTest.grid(row=5, column=1, pady=10)

        self.posPathFile = StringVar()
        self.posPathFile.set("INRIAPerson/Test/pos.lst")
        self.EntryPosTest = Entry(self, textvariable=self.posPathFile)
        self.EntryPosTest.grid(row=5, column=2, pady=10)

        self.negFolderTest = Label(self, text="Negative images folder : ")
        self.negFolderTest.grid(row=6, column=1, pady=10)

        self.negPathFile = StringVar()
        self.negPathFile.set("INRIAPerson/Test/neg.lst")
        self.EntryNegTest = Entry(self, textvariable=self.negPathFile)
        self.EntryNegTest.grid(row=6, column=2, pady=10)

        self.Test = Button(self, text="Test HOG-s", bg="green", fg="white", height=1, width=40,
                           command=lambda: self.createThread("test"))
        self.Test.grid(row=7, column=1, columnspan=2)

        self.OurHog = Label(self, text="Built HOG-descriptor")
        self.OurHog.grid(row=1, column=4)

        self.OpenCvHog = Label(self, text="OpenCv HOG-descriptor")
        self.OpenCvHog.grid(row=1, column=5)

        self.FalsePos = Label(self, text="False Positives : ")
        self.FalsePos.grid(row=2, column=3)
        self.FalseNeg = Label(self, text="False Negatives : ")
        self.FalseNeg.grid(row=3, column=3)
        self.TruePos = Label(self, text="True Positives : ")
        self.TruePos.grid(row=4, column=3)
        self.TrueNeg = Label(self, text="True Negatives : ")
        self.TrueNeg.grid(row=5, column=3)
        self.NumOfTest = Label(self, text="Number of test cases : ")
        self.NumOfTest.grid(row=6, column=3)

        self.OurFPvar = StringVar()
        self.OurFPvar.set("-")
        self.OurFNvar = StringVar()
        self.OurFNvar.set("-")
        self.OurTPvar = StringVar()
        self.OurTPvar.set("-")
        self.OurTNvar = StringVar()
        self.OurTNvar.set("-")
        self.OurNUMvar = StringVar()
        self.OurNUMvar.set("-")
        self.OpenCVFPvar = StringVar()
        self.OpenCVFPvar.set("-")
        self.OpenCVFNvar = StringVar()
        self.OpenCVFNvar.set("-")
        self.OpenCVTPvar = StringVar()
        self.OpenCVTPvar.set("-")
        self.OpenCVTNvar = StringVar()
        self.OpenCVTNvar.set("-")

        self.OurFP = Label(self, textvariable=self.OurFPvar)
        self.OurFP.grid(row=2, column=4)
        self.OurFN = Label(self, textvariable=self.OurFNvar)
        self.OurFN.grid(row=3, column=4)
        self.OurTP = Label(self, textvariable=self.OurTPvar)
        self.OurTP.grid(row=4, column=4)
        self.OurTN = Label(self, textvariable=self.OurTNvar)
        self.OurTN.grid(row=5, column=4)
        self.OurNUM = Label(self, textvariable=self.OurNUMvar)
        self.OurNUM.grid(row=6, column=4, columnspan=2)

        self.OpenCVFP = Label(self, textvariable=self.OpenCVFPvar)
        self.OpenCVFP.grid(row=2, column=5)
        self.OpenCVFN = Label(self, textvariable=self.OpenCVFNvar)
        self.OpenCVFN.grid(row=3, column=5)
        self.OpenCVTP = Label(self, textvariable=self.OpenCVTPvar)
        self.OpenCVTP.grid(row=4, column=5)
        self.OpenCVTN = Label(self, textvariable=self.OpenCVTNvar)
        self.OpenCVTN.grid(row=5, column=5)

        self.labelAccuracy = Label(self, text="Accuracy : ")
        self.labelAccuracy.grid(row=7, column=3)

        self.OurAccuracyVar = StringVar()
        self.OurAccuracyVar.set("-")
        self.OurAccuracy = Label(self, textvariable=self.OurAccuracyVar)
        self.OurAccuracy.grid(row=7, column=4)

        self.OpenCVAccuracyVar = StringVar()
        self.OpenCVAccuracyVar.set("-")
        self.OpenCVAccuracy = Label(self, textvariable=self.OpenCVAccuracyVar)
        self.OpenCVAccuracy.grid(row=7, column=5)

        self.loadingText = StringVar()
        self.loadingText.set("Testing : ")
        self.labelLoading = Label(self, textvariable=self.loadingText)
        self.labelLoading.grid(row=8, column=2, pady=10)
        self.progress = Progressbar(self, length=500, maximum=100000)
        self.progress.grid(row=8, column=3, columnspan=6, pady=10)

        self.loadingText1 = StringVar()
        self.loadingText1.set("Training our model : ")
        self.labelLoading1 = Label(self, textvariable=self.loadingText1)
        self.labelLoading1.grid(row=9, column=2, pady=10)
        self.progress1 = Progressbar(self, length=500, maximum=100000)
        self.progress1.grid(row=9, column=3, columnspan=6, pady=10)

        self.loadingText2 = StringVar()
        self.loadingText2.set("Training OpenCv model : ")
        self.labelLoading2 = Label(self, textvariable=self.loadingText2)
        self.labelLoading2.grid(row=10, column=2, pady=10)
        self.progress2 = Progressbar(self, length=500, maximum=100000)
        self.progress2.grid(row=10, column=3, columnspan=6, pady=10)

        self.svm = cv2.ml.SVM_create()
        self.svmOur = cv2.ml.SVM_create()
        self.winSize = (64, 128)
        self.blockSize = (16, 16)
        self.blockStride = (8, 8)
        self.cellSize = (8, 8)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins)

        # ----------------------------------------------------------------

        # INRIAPerson/Train/pos.lst <-- pos
        # INRIAPerson/Train/neg.lst <-- neg

        # INRIAPerson/Test/pos.lst <-- pos
        # INRIAPerson/Test/neg.lst <-- neg

    def createThread(self, name):
        ThreadedTask(name).start()

    def runThreadTesting(self):
        threadTest = threading.Thread(target=self.test, name='Testing HOG-s')
        threadTest.daemon = True
        threadTest.start()
        threadTest.join()

    def runThreadTrainOur(self):
        threadTrainOurHog = threading.Thread(target=self.trainOur, name='Training Our HOG')
        threadTrainOurHog.daemon = True
        threadTrainOurHog.start()
        threadTrainOurHog.join()

    def runThreadTrainOpenCv(self):
        threadTrainOpenCV = threading.Thread(target=self.train, name='Training OpenCV HOG')
        threadTrainOpenCV.daemon = True
        threadTrainOpenCV.start()
        threadTrainOpenCV.join()

    def trainOur(self):
        global broj1

        if self.EntryPosTrain.get() == "" or self.EntryNegTrain.get() == "":
            return

        self.TrainOurHog['bg'] = 'gray'
        self.TrainOurHog['state'] = 'disabled'

        pos = open(self.EntryPosTrain.get(), "r")
        posList = pos.readlines()
        neg = open(self.EntryNegTrain.get(), "r")
        negList = neg.readlines()

        trainList = []
        trainVectors = []

        x = (len(posList) + len(negList))
        print(x)
        gap = (100000 /  x)

        for el in posList:
            self.progress1['value'] = broj1
            broj1 = broj1 + gap

            trainList.append(1)
            image = cv2.imread("INRIAPerson/" + el.rstrip())
            vectorFin = getProjectHOG(image, False)
            trainVectors.append(vectorFin)

        for el in negList:
            self.progress1['value'] = broj1
            broj1 = broj1 + gap
            trainList.append(-1)
            image = cv2.imread("INRIAPerson/" + el.rstrip())
            vectorFin = getProjectHOG(image, True)
            trainVectors.append(vectorFin)

        sample = []
        for i in range(0, len(trainVectors)):
            sam = []
            for j in range(0, 105):
                for k in range(0, 36):
                    sam.append(trainVectors[i][j][k])
            sample.append(sam)
            #broj1 = broj1 + gap
            self.progress1['value'] = broj1

        self.svmOur.setType(cv2.ml.SVM_C_SVC)
        self.svmOur.setKernel(cv2.ml.SVM_LINEAR)
        self.svmOur.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        self.svmOur.train(np.matrix(sample, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(trainList))
        self.svmOur.save("svm/svmOur.xml")
        self.TrainOurHog['bg'] = 'green'
        self.TrainOurHog['state'] = 'normal'
        self.progress1['value'] = 0
        self.loadingText1.set("Done!")
        time.sleep(5)
        self.loadingText1.set("Training our model : ")
        broj1 = 0

    def train(self):

        global broj2
        if self.EntryPosTrain.get() == "" or self.EntryNegTrain.get() == "":
            return

        self.TrainOpenCvHog['bg'] = 'gray'
        self.TrainOpenCvHog['state'] = 'disabled'

        trainList = []
        trainVectors = []

        pos = open(self.EntryPosTrain.get(), "r")
        posList = pos.readlines()
        neg = open(self.EntryNegTrain.get(), "r")
        negList = neg.readlines()

        gap = 100000 / (len(posList) + len(negList))

        for el in posList:
            self.progress2['value'] = broj2
            broj2 = broj2 + gap

            trainList.append(1)
            image = cv2.imread("INRIAPerson/" + el.rstrip())
            image = preprocessImage(image, False)
            vectorFin = self.hog.compute(img=image)
            vectorFinal = []
            for el in vectorFin:
                vectorFinal.append(el[0])
            trainVectors.append(vectorFinal)

        for el in negList:
            self.progress2['value'] = broj2
            broj2 = broj2 + gap

            trainList.append(-1)
            image = cv2.imread("INRIAPerson/" + el.rstrip())
            image = preprocessImage(image, True)
            vectorFin = self.hog.compute(img=image)
            vectorFinal = []
            for el in vectorFin:
                vectorFinal.append(el[0])
            trainVectors.append(vectorFinal)

        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        self.svm.train(np.array(trainVectors, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(trainList))
        self.svm.save("svm/svmOpenCV.xml")
        self.TrainOpenCvHog['bg'] = 'green'
        self.TrainOpenCvHog['state'] = 'normal'
        self.progress2['value'] = 0
        self.loadingText2.set("Done!")
        time.sleep(5)
        self.loadingText2.set("Training OpenCv model : ")
        broj2 = 0

        return

    def test(self):

        global broj

        if self.EntryPosTest.get() == "" or self.EntryNegTest == "":
            return

        self.Test['bg'] = 'gray'
        self.Test['state'] = 'disabled'

        print("Testing...")

        if (self.textvariable.get() != ""):
            if os.path.isdir(self.textvariable.get()):
                self.svm = self.svm.load(self.textvariable.get() + "/" + "svmOpenCV.xml")
                self.svmOur = self.svmOur.load(self.textvariable.get() + "/" + "svmOur.xml")

        posTest = open(self.EntryPosTest.get(), "r")
        testList = posTest.readlines()
        negTest = open(self.EntryNegTest.get(), "r")
        negTestList = negTest.readlines()

        for el in negTestList:
            testList.append(el)

        falsePositives = 0
        falseNegatives = 0
        truePositives = 0
        trueNegatives = 0
        gap = (100000 / len(testList)) / 2
        testedImageCount = 0
        for im in testList:
            self.progress['value'] = broj
            broj = broj + gap

            image = cv2.imread("INRIAPerson/" + im.rstrip())
            if "pos" in im:
                if image is not None:
                    image = preprocessImage(image, False)
                    testedImageCount += 1
            else:
                if image is not None:
                    image = preprocessImage(image, True)
                    testedImageCount += 1
            if image is not None:
                vectorFin = self.hog.compute(img=image)
                vectorFin = vectorFin.tolist()
                vectorFinal = []
                for el in vectorFin:
                    vectorFinal.append(el[0])
                sam = vectorFinal
                samMat = np.matrix(sam, dtype=np.float32)
                response = self.svm.predict(samMat)[1]
                if response == 1.0:
                    if "pos" in im:
                        truePositives += 1
                    else:
                        falsePositives += 1
                elif response == -1.0:
                    if "neg" in im:
                        trueNegatives += 1
                    else:
                        falseNegatives += 1
                else:
                    imageShow = cv2.resize(imageShow, (300, 300), imageShow)
                    cv2.imshow("Something else", imageShow)
                    cv2.waitKey(0)

        self.OurNUMvar.set(testedImageCount)
        self.OpenCVFPvar.set(falsePositives)
        self.OpenCVFNvar.set(falseNegatives)
        self.OpenCVTPvar.set(truePositives)
        self.OpenCVTNvar.set(trueNegatives)
        self.OpenCVAccuracyVar.set((truePositives + trueNegatives) / len(testList))

        falsePositives = 0
        falseNegatives = 0
        truePositives = 0
        trueNegatives = 0

        for im in testList:

            self.progress['value'] = broj
            broj = broj + gap

            imageTest = cv2.imread("INRIAPerson/" + im.strip())
            if imageTest is not None:
                if "pos" in im:
                    vectorFin = getProjectHOG(imageTest, False)
                else:
                    vectorFin = getProjectHOG(imageTest, True)

                sam = []
                for j in range(0, 105):
                    for k in range(0, 36):
                        sam.append(vectorFin[j][k])
                samMat = np.matrix(sam, dtype=np.float32)
                response = self.svmOur.predict(samMat)[1]
                if response == 1.0:
                    if "pos" in im:
                        truePositives += 1
                    else:
                        falsePositives += 1
                elif response == -1.0:
                    if "neg" in im:
                        trueNegatives += 1
                    else:
                        falseNegatives += 1
                else:
                    imageShow = cv2.resize(imageShow, (300, 300), imageShow)
                    cv2.imshow("Something else", imageShow)
                    cv2.waitKey(0)

        self.OurFPvar.set(falsePositives)
        self.OurFNvar.set(falseNegatives)
        self.OurTPvar.set(truePositives)
        self.OurTNvar.set(trueNegatives)
        self.OurAccuracyVar.set((truePositives + trueNegatives) / len(testList))
        self.Test['bg'] = 'green'
        self.Test['state'] = 'normal'
        self.progress['value'] = 0
        self.loadingText.set("Done!")
        time.sleep(5)
        self.loadingText.set("Testing : ")
        broj = 0
        return


class ThreadedTask(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.name = name

    def run(self):
        if self.name == "test":
            my_gui.runThreadTesting()
        elif self.name == "trainOur":
            my_gui.runThreadTrainOur()
        elif self.name == "trainOpenCv":
            my_gui.runThreadTrainOpenCv()


root = Tk()
my_gui = HOGdescriptor(root)
root.mainloop()

