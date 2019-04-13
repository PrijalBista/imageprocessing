import cv2
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np


def showImage(info,image):
    cv2.imshow(info,image)
    cv2.waitKey(0)
    cv2.destroyWindow(info)

def showTwoImages(inp,out):
    cv2.imshow("Input Image",inp)
    cv2.imshow("Output Image",out)
    cv2.waitKey(0)
    cv2.destroyWindow("Input Image")
    cv2.destroyWindow("Output Image")
    
def binarize(img,threshold):
    imgcpy=img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j]<=threshold):
                imgcpy[i,j]=255
            else:
                imgcpy[i,j]=0
    return imgcpy

def negative(image):
    maxval=image.max()
    return (maxval-image)

def getFilter(name):
    mean = (1/9)*np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])
    weightedmean=np.array([
        [1/16,2/16,1/16],
        [2/16,4/16,2/16],
        [1/16,2/16,1/16]
    ])
    
    highpass = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]
    ])
    
    gauss = (1/57)*np.array([
    [0,1,2,1,0],
    [1,3,5,3,1],
    [2,5,9,5,2],
    [1,3,5,3,1],
    [0,1,2,1,0]
    ])

    laplacian = (1/16)*np.array([
        [0,0,-1,0,0],
        [0,-1,-2,-1,0],
        [-1,-2,16,-2,-1],
        [-1,-2,16,-2,-1],
        [0,0,-1,0,0]
    ])

    box = (1/9)*np.array([
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ])
    
    
    return {
        'MEAN': mean,
        'WEIGHTEDMEAN': weightedmean,
        'HIGHPASS': highpass,
        'BOX':box,
        'GAUSSIAN':gauss,
        'LAPLACIAN':laplacian
    }.get(name,mean)

#correlation
def correlation(input,filter):
    sum=0
    for i in range(filter.shape[0]):
        for j in range(filter.shape[0]):
            sum +=input[i,j]*filter[i,j]
    middle = int((filter.shape[0]+1)/2)
    output = np.copy(input)
    output[middle-1,middle-1]=int(round(sum))
    return output

def maxcorrelation(mat):
    matrix=mat.copy()
    middle = int((matrix.shape[0]+1)/2)
    matrix[middle-1,middle-1]=matrix.max()
    return matrix

def mincorrelation(mat):
    matrix=mat.copy()
    middle = int((matrix.shape[0]+1)/2)
    matrix[middle-1,middle-1]=matrix.min()
    return matrix
    
def mediancorrelation(mat):
    matrix=mat.copy()
    middle = int((matrix.shape[0]+1)/2)
    matrix[middle-1,middle-1]=np.median(matrix)
    return matrix
 
#convolution
from scipy import ndimage
def convolution(input,filter):
    return ndimage.convolve(input,filter)

def smoothingCorrelation(img,fltertype="MEAN",padding=1): #padding vaneko 0 padding default ma 1 ho
    #showImage('orig',img)
    if(fltertype=='LOWPASS'):
        flter = getFilter('HIGHPASS')
    else:
        flter=getFilter(fltertype)
    imgcpy=img.copy()
    movement_vertical = imgcpy.shape[0]-flter.shape[0]
    movement_horizontal = imgcpy.shape[1]-flter.shape[1]
    #print("row ma :",movement_vertical,"column ma",movement_horizontal)
    for i in range(movement_vertical):
        for j in range (movement_horizontal):
            mat1 = img[i:i+flter.shape[0], j:j+flter.shape[0]]
            #matbkup=mat1.copy()
            if(fltertype=='MAX'):
                imgcpy[i:i+flter.shape[0], j:j+flter.shape[0]]=maxcorrelation(mat1)
                
            elif(fltertype=='MIN'):
                imgcpy[i:i+flter.shape[0], j:j+flter.shape[0]]=mincorrelation(mat1)
                
            elif(fltertype=='MEDIAN'):
                imgcpy[i:i+flter.shape[0], j:j+flter.shape[0]]=mediancorrelation(mat1) 
            else:
                imgcpy[i:i+flter.shape[0], j:j+flter.shape[0]]=correlation(mat1,flter)
            #print(matbkup,"op",imgcpy[i:i+flter.shape[0], j:j+flter.shape[0]],"\n")
    if(fltertype=='LOWPASS'):
        return (img-imgcpy)
    return imgcpy

def smoothingConvolution(img,fltertype='MEAN'):
    if(fltertype=='LOWPASS'):
        flter=getFilter('HIGHPASS')
        hp=convolution(img,flter)
        return (img-hp)
    elif(fltertype=='HIGHBOOST'):
        flter=getFilter('HIGHPASS')
        hp=convolution(img,flter)
        return (img+hp)
    else:
        flter=getFilter(fltertype)
        return convolution(img,flter)

def histogram(image):
    hist = np.zeros((256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a = image[i,j]
            hist[a] += 1
            int(hist[a])
    
    return hist.astype(np.int)

def cumulativeHistogram(hist):
    cum_hist = hist.copy()
    
    for i in np.arange(1,256):
        cum_hist[i]=cum_hist[i]+cum_hist[i-1]
    
    return cum_hist.astype(np.int)

def plothistogram(image):
    hist=histogram(image)
    plt.bar(np.arange(0,256),hist)
    plt.get_current_fig_manager().window.raise_()
    plt.show()

def histogramEqualization(image):
    hist= histogram(image)
    cum_hist= cumulativeHistogram(hist)
    totalpixels = image.shape[0]*image.shape[1]
    imgcpy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            o = image[i,j]
            new = cum_hist[o]*255/totalpixels
            imgcpy[i,j]=new
    return imgcpy

def histogramMatching(image,ref_img):
    pixels_ref = ref_img.shape[0]*ref_img.shape[1]
    pixels = image.shape[0]*image.shape[1]
    
    hist= histogram(image)
    hist_ref = histogram(ref_img)
    
    cum_hist = cumulativeHistogram(hist)
    cum_hist_ref= cumulativeHistogram(hist_ref)
    
    cum_hist_prob = cum_hist/pixels
    cum_hist_ref_prob = cum_hist_ref/pixels_ref
    
    imgcpy = image.copy()
    
    new = np.zeros(256)
    
    for a in range(256):
        j = 255;
        while True:
            new[a]= j
            j= j-1
            if j<0 or cum_hist_prob[a]>cum_hist_ref_prob[j]:
                break

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            old = image[i,j]
            imgcpy[i,j]=new[old]
    
    return imgcpy

def linearFilter(image,fltertype):
    flter = getFilter(fltertype)
    imcpy = image.copy()
    for i in np.arange(2,image.shape[0]-2):
        for j in np.arange(2,image.shape[1]-2):
            sum=0
            for k in np.arange(-2,3):  #-2,-1,0,1,2
                for l in np.arange(-2,3):
                    a =image[i+k,j+l] #neighbours ko value
                    p= flter[k+2,l+2]
                    sum += p*a
            
            imcpy[i,j]=sum
    
    return imcpy

#global declarations
#filename = 'videoKoTest/opencv_frame_1.png'
filename = 'images/cat.png'
originalimage = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
inputimage = originalimage.copy()
processedimage = originalimage.copy()

def selectfile():
    filename = input("Please enter the pathname for the image : ")
    print(filename)
    global originalimage,inputimage,processedimage
    originalimage = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    inputimage = originalimage.copy()
    processedimage = originalimage.copy()
    showImage("Selected Image", inputimage)

def selectImage():

    ip = int(input("On which image 1.original 2.Previously processed image"))
 
    global inputimage,originalimage1
    if(ip ==1):
        inputimage=originalimage
    else:
        inputimage=processedimage


def linearFilters():
    print("Choose the linear filter u wanna apply")
    print("1.Box filter\n2.Gaussian filter\n3.Laplacian filter")
    ch = int(input("enter your choice : "))
    selectImage()    
    global processedimage
    if(ch==1):
        processedimage = linearFilter(inputimage,'BOX')
        showTwoImages(inputimage,processedimage)
    elif(ch==2):
        processedimage = linearFilter(inputimage,'GAUSSIAN')
        showTwoImages(inputimage,processedimage)
    elif(ch==3):
        processedimage = linearFilter(inputimage,'LAPLACIAN')
        showTwoImages(inputimage,processedimage)
    else:
        return

def nonLinearFilters():
    print("List of non Linear Filters")
    print("1.MIN\n2.MAX\n3.MEAN\n4.WEIGHTEDMEAN\n5.MEDIAN")
    print("6.HIGHPASS\n7.LOWPASS\n8.HIGHBOOST")
    ch = int(input("enter your choice"))
    selectImage()
    global processedimage
    if(ch==1):
        processedimage = smoothingConvolution(inputimage,fltertype='MIN')
        showTwoImages(inputimage,processedimage)
    elif(ch==2):
        processedimage = smoothingConvolution(inputimage,fltertype='MAX')
        showTwoImages(inputimage,processedimage)
    elif(ch==3): 
        processedimage = smoothingConvolution(inputimage,fltertype='MEAN')
        showTwoImages(inputimage,processedimage)
    elif(ch==4):
        processedimage = smoothingConvolution(inputimage,fltertype='WEIGHTEDMEAN')
        showTwoImages(inputimage,processedimage)
    elif(ch==5):
        processedimage = smoothingConvolution(inputimage,fltertype='MEDIAN')
        showTwoImages(inputimage,processedimage)
    elif(ch==6):
        processedimage = smoothingConvolution(inputimage,fltertype='HIGHPASS')
        showTwoImages(inputimage,processedimage)
    elif(ch==7):
        processedimage = smoothingConvolution(inputimage,fltertype='LOWPASS')
        showTwoImages(inputimage,processedimage)
    elif(ch==8):
        processedimage = smoothingConvolution(inputimage,fltertype='HIGHBOOST')
        showTwoImages(inputimage,processedimage)
    else:
        return

def histogramTransformations():
    print("Some of histogram majorly used transformations")
    print("1.Equalization\n2.Histogram Matching (reference image required)")
    ch= int(input("enter your choice"))
    selectImage()
    global processedimage
    if(ch==1):
        processedimage=histogramEqualization(inputimage) 
        showTwoImages(inputimage,processedimage)
        plt.subplot(221)
        plt.bar(np.arange(0,256),histogram(inputimage))
        plt.subplot(222)
        plt.bar(np.arange(0,256),histogram(processedimage))
        plt.subplot(223)
        plt.bar(np.arange(0,256),cumulativeHistogram(histogram(inputimage)))
        plt.subplot(224)
        plt.bar(np.arange(0,256),cumulativeHistogram(histogram(processedimage)))
        plt.show()
    if(ch==2):
        fn = input("enter the filename of the reference image ")
        refimage = cv2.imread(fn,cv2.IMREAD_GRAYSCALE)
        processedimage=histogramMatching(inputimage,refimage)
        #showTwoImages(inputimage,processedimage)
        cv2.imshow("Input",inputimage)
        cv2.imshow("Reference",refimage)
        cv2.imshow("Output",processedimage)
        plt.subplot(311)
        plt.bar(np.arange(0,256),cumulativeHistogram(histogram(inputimage)))
        plt.title("original histogram")
        plt.subplot(312)
        plt.bar(np.arange(0,256),cumulativeHistogram(histogram(refimage)))
        plt.title("reference's histogram")
        plt.subplot(313)
        plt.bar(np.arange(0,256),cumulativeHistogram(histogram(refimage)))
        plt.title("matched  histogram")
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

flag = True
while(flag):
    print("Image Processing Project")
    print("Prijal Bista 113/BCT/072")
    print("1.Select file\n2.Histogram\n3.Exit\n4.Negative")
    print("5.Linear Filters\n6.Non Linear Filters\n7.Histogram Transformations")
    choice = int(input("enter your choice : "))

    if(choice==3):
        print("exiting...")
        flag=False
    elif(choice==1):
        selectfile()
    elif(choice == 2):
        print("histogram")
        plothistogram(inputimage)
    elif(choice==4):
        showTwoImages(inputimage,negative(inputimage))
    elif(choice==5):
        linearFilters()
    elif(choice==6):
        nonLinearFilters()
    elif(choice==7):
        histogramTransformations()




