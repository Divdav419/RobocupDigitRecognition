import numpy as np
import cv2
from PIL import Image
import sys

SMALL_WIDTH = 100  # width of small squares in small()
SMALL_HEIGHT = 100  # height of small squares in small()
binaryTresh = 190  # threshhold of what to count as black
max_gap = 10  # no. of gaps allowed to still count it as one entity


# cap = cv2.VideoCapture(0)


def areaFilter(minArea, inputImage):
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, 4)

    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage


# returns frame in black and white
def bw_frame(cap):
    ret, frame = cap.read()
    # frame = cap

    # get K channel:
    imgFloat = frame.astype(np.float64) / 255
    kChannel = 1 - np.max(imgFloat, axis=2)
    # convert to unit 8
    kChannel = (255 * kChannel).astype(np.uint8)

    # Threshold
    _, binaryImage = cv2.threshold(kChannel, binaryTresh, 255, cv2.THRESH_BINARY)

    # filter
    minArea = 100
    binaryImage = areaFilter(minArea, binaryImage)

    kernelSize = 3
    opIterations = 2
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                                   cv2.BORDER_REFLECT101)

    # resize image
    binaryImage = Image.fromarray(binaryImage)
    binaryImage = binaryImage.resize((100, 100), Image.LANCZOS).convert('RGB')
    binaryImage = np.array(binaryImage)

    # cv2.imshow('Object Recognition', binaryImage)      # shows image in new window
    return binaryImage
    # if cv2.waitKey(1) & 0xFF == ord('q'): break


# creates list to describe graph (every element is a list that consists of all connected nodes, index of elements
# define the node in question)
def create_graph(frame):
    height, width = frame.shape[:2]
    #print(width, height)
    l = []
    for i in range(width * height):
        sl = []
        # add left and right if not on sides
        if i % width != 0: sl.append(i - 1)
        if i % width != width - 1: sl.append(i + 1)
        # add up and down
        if i >= width: sl.append(i - width)
        if i < width * (height - 1): sl.append(i + width)
        # add diagonals
        if i % width != 0 and i >= width: sl.append(i - width - 1)
        if i % width != 0 and i < width * (height - 1): sl.append(i + width - 1)
        if i % width != width - 1 and i >= width: sl.append(i - width + 1)
        if i % width != width - 1 and i < width * (height - 1): sl.append(i + width + 1)
        l.append(sl)
    return l


gap = 0
checked = []  # list of all checked nodes


# coords = [0, 10, 0, 10] # max_x, min_x, max_y, min_y
def search(frame, graph, node_index, coords):  # looks for all connected white nodes from given node
    if node_index not in checked:
        global gap
        # global coords

        height, width = frame.shape[:2]

        x = node_index % width
        y = int(node_index / width)

        if (frame[y, x] == 255).all(): # or gap <= max_gap:
            # print(x, y)
            coords = [max(coords[0], x), min(coords[1], x), max(coords[2], y), min(coords[3], y)]  # update coordinates
            if not (frame[y, x] == 255).all():
                gap += 1  # add gap if pixel is not white
            else:
                gap = 0
            checked.append(node_index)  # add it to the list of checked nodes

            # search neighboring nodes
            for i in graph[node_index]:
                coords = search(frame, graph, i, coords)
    return coords


def find_connected(cap):  # finds all white groups in a frame
    global gap
    # global coords
    global checked

    structs = []  # list of 4 coordinates of all structures

    # print("before bw")
    frame = bw_frame(cap)
    # print("after bw")
    # frame = cap
    height, width = frame.shape[:2]
    sys.setrecursionlimit(width * height + 1)

    graph = create_graph(frame)
    # print("after graph")

    for i in range(len(graph)):
        x = i % width
        y = int(i / width)
        if (frame[y, x] == 255).all() and i not in checked:  # if pixel is white and has not been visited before
            gap = 0
            # print("before search")
            structs.append(search(frame, graph, i, [0, width, 0, height]))
    checked = []

    return frame, structs


# creates a square outline of object
# img = np.zeros((512,512,3), np.uint8)
def outline(frame, sides):
    cv2.rectangle(frame, (sides[1], sides[2]), (sides[0], sides[3]), (0, 255, 0), 1)
    # print('--------------------------------------------------------------------------------------')
    # return (sides[0]-sides[1]) * (sides[2]-sides[3])


# outline(img, 384, 510, 0, 128)
# print("Black:", img[0, 0]) # img[y, x] returns rgb value at [x, y]
# print("Green:", img[0, 510])
# print((img[0, 510] == 0).all())
# if (img[0, 0] == 0).all(): print("It is black")
# cv2.imshow("rect", img)
# cv2.waitKey(0)


'''def small(frame, x, y):
    cv2.imshow('Smaller Image', frame[x : x + SMALL_WIDTH, y : y + SMALL_HEIGHT])
    #height, width = frame.shape[:2]'''

'''
Find biggest white blob and put a square around it
- Osszefuggest kereso graf:
    - see yellow opaque fuzet
- find smallest/greatest x and smallest/greatest y and make a square
'''

'''
IMPORTANT!!!
Need to reduce size of images in bw_frame(): https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array

            from LIP import Image
            
            pic = None
            img = pic.putdata(frame) OR Image.fromarray(frame)
            
            img = img.resize((200, 200), Image.LANCZOS)
            frame = np.array(img)

resizes image
'''
