import cv2
import re 


def replaceInclude(fileName:str, finalFileName:str):
    # Read the contents of main.c into a string
    with open(fileName, 'r') as file:
        content = file.read()

    # Find all occurrences of the pattern '#include "filename.h"'
    matches = re.findall(r'#include "(.+?\.h)"', content)

    # Loop through each match and replace it with the contents of the corresponding file
    for match in matches:
        with open(match, 'r') as file:
            file_content = file.read()
            content = content.replace(f'#include "{match}"', file_content)

    # Write the updated content back to main.c
    with open(finalFileName, 'w') as file:
        file.write(content)


def printToArrayFile(fileName:str, arrayName:str):

    im = cv2.imread(fileName)

    im = cv2.cvtColor(im,cv2.COLOR_BGRA2RGB)

    resolutionX = len(im[0])
    resolutionY = len(im)

    red = list(range(len(im)))
    green = list(range(len(im)))
    blue = list(range(len(im)))

    for y in range(len(im)):
        red[y] = list(range(len(im[0])))
        green[y] = list(range(len(im[0])))
        blue[y] = list(range(len(im[0])))
        for x in range(len(im[0])):
            redNum = im[y][x][0]
            redNum = int(redNum/255 * 31)

            greenNum = im[y][x][1]
            greenNum = int(greenNum/255 * 63)

            blueNum = im[y][x][2]
            blueNum = int(blueNum/255 * 31)

            red[y][x] = hex(redNum)
            green[y][x] = hex(greenNum)
            blue[y][x] = hex(blueNum)

    pixel = list(range(len(im)))
    for y in range(len(im)):
        pixel[y] = list(range(len(im[0])))
        for x in range(len(im[0])):
            pixel[y][x] = hex(0)
            pixel[y][x] = int(pixel[y][x], 16) | int(red[y][x], 16)
            pixel[y][x] <<= 6
            pixel[y][x] = pixel[y][x] | int(green[y][x], 16)
            pixel[y][x] <<= 5
            pixel[y][x] = pixel[y][x] | int(blue[y][x], 16)
            pixel[y][x] = hex(pixel[y][x])

    with open('graphicHeaders/' + arrayName + '.h', 'w') as file:
        print('short int ' + arrayName + '[' + str(resolutionY) + '][' + str(resolutionX) + '] = {', file=file, end='')
        for y in range(resolutionY):
            print('{', file=file, end='')
            for x in range(resolutionX):
                if(x < resolutionX - 1):
                    print(str(pixel[y][x]) + ',', file=file, end='')
                else:
                    print(str(pixel[y][x]), file=file, end='')

            if(y < resolutionY-1):
                print('},', file=file, end='')
            else:
                print('}', file=file, end='')
        print('};', file=file, end='')

    print('Complete')

# printToArrayFile('graphics/button.png', 'button')
# printToArrayFile('graphics/buttonClick.png', 'buttonClick')
replaceInclude('graphics.c', 'updatedGraphics.c')