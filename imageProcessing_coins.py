# from ast import main
# from distutils.command.clean import clean
# from random import gauss
import cv2
from cv2 import imread
from matplotlib import contour
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

# Read image formatted with .JPG
def readImage(path):
    
    return cv2.imread(path)

# Shows the image
def showImage(title, image):
    cv2.imshow(title, image)


# We want the new image to be 100% of the original image
def resizeImage(image):
    height = image.shape[0]
    width = image.shape[1]
    scale_factor = 3
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    dimensions = (new_width, new_height)
   
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

# Apply Gaussian blurring (Gaussian Blur()) to do noise filtering and locate the most prominent edges of the smoothed image
def imageGaussianBlur(image):
    return cv2.GaussianBlur(image, (1, 1), cv2.BORDER_DEFAULT)

# Image enhancement converting to grayscale and then binarizing
def imageEnhancement(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    ret, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)

    return thresholded

#Clean the image by removing noise
def imageCleaning(image): 
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3 , 3)))
    

# Detect the edges with Canny and Look for external contours inside the image
def getContours(image):
    borders = cv2.Canny(image, 50, 700)
    (contours, _) = cv2.findContours(borders,
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

# Shows the number of contours
def printNumberOfContours(contours):
    print("I have found {} contours".format(len(contours)))

# Calculate the areas, perimeters and equivalent diameters of irregular figures
def getAreasPerimEquiDiam(contours):
    all_areas = []
    all_perimeters = []
    all_equi_diameters = []

    for contour in contours:
        area = cv2.contourArea(contour)  
        perimeter = cv2.arcLength(contour, True)  
        equi_diameter = np.sqrt((4 * area) / np.pi)
        all_areas.append(area)
        all_perimeters.append(perimeter)
        all_equi_diameters.append(equi_diameter)

    return all_areas, all_perimeters, all_equi_diameters

# Gets a specific parameter
def getMedidas(tipo, contours):
    all_areas = []
    all_perimeters = []
    all_equi_diameters = []
    for contour in contours:
        if (tipo == "area"):
            area = cv2.contourArea(contour)  
            all_areas.append(area)
            return all_areas
        if (tipo == "perimeter"):
            perimeter = cv2.arcLength(contour, True)  
            all_perimeters.append(perimeter)
            return all_perimeters
    
        equi_diameter = np.sqrt((4 * area) / np.pi)
        all_equi_diameters.append(equi_diameter)
        return all_equi_diameters

# Export data to excel
def exportAreasToExcel(areas, excel_name):
    areas_df = pd.DataFrame(areas)
    areas_df.to_excel(excel_name + ".xlsx")

def exportPerimsToExcel(perimeters, excel_name):
    perimeters_df = pd.DataFrame(perimeters)
    perimeters_df.to_excel(excel_name + ".xlsx")

def exportEquiDiametersToExcel(equi_diameters, excel_name):
    equi_diameters_df = pd.DataFrame(equi_diameters)
    equi_diameters_df.to_excel(excel_name + ".xlsx")    

#To show the asociation in the console
def getAreasPerimetrosPerimetroEqui_withTags(contours):
    all_areas = []
    all_perimeters = []
    all_diameters_equi = []

    cont = 0
    for contour in contours:
        area = cv2.contourArea(contour)  
        perimeter = cv2.arcLength(contour, True)  
        equi_diameter = np.sqrt((4 * area) / np.pi)
        cont += 1
        all_areas.append({'id': cont, 'area': area})
        all_perimeters.append({'id': cont, 'perimeter': perimeter})
        all_diameters_equi.append({'id': cont, 'equivDiameter': equi_diameter})

    return all_areas, all_perimeters, all_diameters_equi

def addTags(image, contours):
    cont = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cont += 1
        # Text characteristics
        text = str(cont)
        location = (x, y)
        font = cv2.FONT_HERSHEY_DUPLEX
        fontZise = 1
        fontColor = (102, 204, 0)
        tickness = 1

        cv2.putText(image, text, location, font, fontZise,
                    fontColor,    tickness, bottomLeftOrigin=False)

# Main function that executes the other functions
def main():
    original_image = readImage(".\Images\Coins.jpg")
    # resized_image = resizeImage(original_image)
    gauss = imageGaussianBlur(original_image)
    enhaced_image = imageEnhancement(gauss)
    cleaned_image = imageCleaning(enhaced_image)


    contours = getContours(enhaced_image) 
    # areas, perims, ed = getAreasPerimEquiDiam(contours)
    # Mostramos el número de contornos
    print("I have found {} contours".format(len(contours)))
    # print ("All the areas: ", areas) 
    # print ("All the perimeters: ", perims)
    # print ("All the equiv diameters: ", ed) 

    addTags(original_image, contours)
    areas, perimeters, equivDiameters = getAreasPerimetrosPerimetroEqui_withTags(contours)

    print('============= Areas =============')
    for a in areas:
        print(a, '\n')
    print('============= Perimeters =============')
    for p in perimeters:
        print(p, '\n')
    # print(perimetros)
    print('============= Equivalent Diameters =============')
    for ed in equivDiameters:
        print(ed, '\n')

    #exportAreasToExcel(areas, "areas_output Coins")
    # exportPerimsToExcel(perims, "perimeters_output Coins" )
    # exportEquiDiametersToExcel(ed, "diameters_output Coins")

    #Shows all images
    # showImage("Resized Image", resized_image)
    showImage("Smoothed Image", gauss)
    showImage("Thresholded Image", enhaced_image)
    showImage("Cleaned Image", cleaned_image)
   
    # Draw the cleaned image with contours
    cv2.drawContours(cleaned_image, contours, -1, (0,255,0), 5)
    showImage("Cleaned image with Contours", cleaned_image)

    # Draw the original image with green contours
    cv2.drawContours(original_image, contours, -1, (0,255,0), 3)
    showImage("Original coins image with draw Contours", original_image)

    
    






    # h, w = enhaced_image[:2]
    # h = enhaced_image.shape[0]
    # w = enhaced_image.shape[1]

    # img_floodFill = enhaced_image.copy()
    # img_floodFill = resized_image.copy()
    # mask = np.zeros((h+2, w+2), np.uint8)
    # cv2.floodFill(img_floodFill, mask, (0,0), (0,0,255))

    # connectivity = 4
    # flags = connectivity
    # flags |= cv2.FLOODFILL_FIXED_RANGE

    # cv2.floodFill(enhaced_image, None, (0,0), (0, 255, 255),
    #               (1,) * 3, (1,) * 3, flags)
    # cv2.imshow('relleno', enhaced_image)

    # showImage("Mask", img_floodFill)

    # for c in contornos:
    #     cv2.fillPoly(resized_image, pts=[c], color=(0, 0, 255), lineType=cv2.FILLED)

    # showImage("test", resized_image)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
