import cv2

im = cv2.imread("graphics/titleBackground.png")
im = cv2.cvtColor(im,cv2.COLOR_BGRA2RGB)

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
        greenNum = int(greenNum/255 * 64)

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
        pixel[y][x] |= red[y][x]
        pixel[y][x] <<= 6
        pixel[y][x] |= green[y][x]
        pixel[y][x] <<= 5
        pixel[y][x] |= blue[y][x]

print(pixel)