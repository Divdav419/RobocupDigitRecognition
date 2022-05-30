import numpy as np
import cv2
import blackAndWhite as bw
import Letters

PRECISION = 0.65    # What certainty it counts as certain for a matching letter
MIN_SIZE = 5    # minimum size of edge of struct

letters = {
    0: "Not a letter",
    1: "H",
    2: "S",
    3: "U"
}

cap1 = cv2.VideoCapture(0)

while True:
    frame, structs = bw.find_connected(cap1)
    for struct in structs:
        if struct[1] >= struct[2] + MIN_SIZE and struct[3] >= struct[4] + MIN_SIZE:
            # print(struct)
            struct_frame = frame[struct[4]:struct[3], struct[2]:struct[1]]
            l, certainty = Letters.image_matching(struct_frame, PRECISION)
            if l != 0 or certainty < 0.5:
                # print(letters[l]+":", certainty)
                cv2.putText(frame, (letters[l]+": " + str(int(certainty*100)) + "%"), (struct[2], struct[4]),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, 2)
            bw.outline(frame, struct[1:])
    
    cv2.imshow("Outlined", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

'''
img = cv2.imread("ToPixelate\PICT0058.jpg")
frame, structs = bw.find_connected(img)
print(structs)
for struct in structs:
    #print("(", struct[0], ";", struct[2], "), (", struct[1], ";", struct[3], ")")
    bw.outline(frame, struct)

cv2.rectangle(frame, (1, 1), (99, 99), (0, 255, 0), 2)
cv2.imshow("outlined", frame)
cv2.waitKey(0)'''