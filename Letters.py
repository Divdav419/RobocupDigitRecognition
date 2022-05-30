import numpy as np
import cv2
import blackAndWhite as bw
from PIL import Image

# Images of letters (0 = black; 1 = white); H, S, U
l_img = [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]],\
		[[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]],\
		[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]]

def r(num): return int(num) if (num - int(num)) < 0.5 else (int(num) + 1)	# rounds number


def letter_array(img):
	# create array with 0/1 for each pixel
	out_img = []

	# check each pixel if it is white or black
	for x in img:
		row = []
		for y in x:
			w = False
			for z in y:
				if z > 100:
					w = True
					break
			if w:
				row.append(1)
			else:
				row.append(0)
		out_img.append(row)

	return out_img  # width = letter_array()[0]; height = letter_array()


def map_pixel(x, y, n, o):	# maps pixel (x;y) from image n to image o; returns (x, y)
	wn = len(n[0])
	hn = len(n)
	wo = len(o[0])
	ho = len(o)

	return int((x+0.5) * (wo/wn)), int((y+0.5) * (ho/hn))


def image_matching(img, precision):		# returns letter (0=none, 1=h, 2=s, 3=u); percent matched
	img = letter_array(img)

	# dimensions of img
	w = len(img[0])
	h = len(img)

	# certainty for each letter
	averages = [0, 0, 0]

	# iterate through img
	for y in range(h):
		for x in range(w):
			for i in range(3):
				xm, ym = map_pixel(x, y, img, l_img[i])
				# add to matches[] if color matches
				averages[i] += int(l_img[i][ym][xm] == img[y][x])

	# return highest average and its letter
	highest = max(averages)
	if highest < precision * (w*h):
		return 0, (highest / (w*h))
	else:
		for i in range(3):
			if highest == averages[i]:
				return i+1, (highest / (w*h))

'''
img = cv2.imread("S_test.png")
xc = -1
yc = -1
zc = -1
for x in img:
	xc += 1
	yc = -1
	for y in x:
		yc += 1
		zc = -1
		for z in y:
			zc += 1
			if z < 100:
				img[xc, yc, zc] = 0
			else:
				img[xc, yc, zc] = 255
#cv2.imshow("img", img)
#cv2.waitKey(0)

print(image_matching(img, 0.75))
'''
'''
h_test = [[0 for x in range(len(l_img[0][0]))] for y in range(len(l_img[0]))]
for y in range(len(l_img[0])):
	for x in range(len(l_img[0][0])):
		if l_img[0][y][x] == 0:
			h_test[y][x] = [0, 0, 0]
		else:
			h_test[y][x] = [255, 255, 255]

s_test = [[0 for x in range(len(l_img[1][0]))] for y in range(len(l_img[1]))]
for y in range(len(l_img[1])):
	for x in range(len(l_img[1][0])):
		if l_img[1][y][x] == 0:
			s_test[y][x] = [0, 0, 0]
		else:
			s_test[y][x] = [255, 255, 255]

# print(map_pixel(3, 6, s_test, h_test))
# print(map_pixel(2, 3, s_test, h_test))
print(image_matching(h_test, 0.75))
print(image_matching(s_test, 0.75))
'''
'''
def image_match(img, precision=0.75):	# returns letter (0=none, 1=h, 2=s, 3=u); precent matched
	img = letter_array(img)

	# dimensions of img
	w1 = len(img[0])
	h1 = len(img)
	# dimensions of letters
	wh = len(l_img[0][0])
	hh = len(l_img[0])
	ws = len(l_img[1][0])
	hs = len(l_img[1])
	wu = len(l_img[2][0])
	hu = len(l_img[2])

	# create matching arrays
	h_matches = [[0 for x in range(wh)] for y in range(hh)]
	s_matches = [[0 for x in range(ws)] for y in range(hs)]
	u_matches = [[0 for x in range(wu)] for y in range(hu)]

	# iterate through img
	for y in range(len(img)):
		for x in range(len(img[0])):
			# compare
			xh = r((wh/w1)*x)
			yh = r((hh/h1)*y)

			xs = r((ws/w1)*x)
			ys = r((hs/h1)*y)

			xu = r((wu/w1)*x)
			yu = r((hu/h1)*y)

			# save checked letters
			h_matches[yh][xh] += int(img[y][x] == l_img[0][yh][xh])
			s_matches[ys][xs] += int(img[y][x] == l_img[0][ys][xs])
			u_matches[yu][xu] += int(img[y][x] == l_img[0][yu][xu])

	# iterate through matches to find averages
	h_average = 0
	s_average = 0
	u_average = 0

	for y in range(hh):
		for x in range(wh):
			h_matches[y][x] = h_matches[y][x]/((r(w1/(2*wh))+1) * (r(h1/(2*hh))+1)) # finds average
			h_average += h_matches[y][x]
	h_average = h_average/(hh*wh)

	for y in range(hs):
		for x in range(ws):
			s_matches[y][x] = s_matches[y][x]/((r(w1/(2*ws))+1) * (r(h1/(2*hs))+1)) # finds average
			s_average += s_matches[y][x]
	s_average = s_average/(hs*ws)

	for y in range(hu):
		for x in range(wu):
			u_matches[y][x] = u_matches[y][x]/((r(w1/(2*wu))+1) * (r(h1/(2*hu))+1)) # finds average
			u_average += u_matches[y][x]
	u_average = u_average/(hu*wu)

	greatest = max(h_average, s_average, u_average)
	if greatest < precision:
		return 0, None
	elif greatest == h_average:
		return 1, h_average
	elif greatest == s_average:
		return 2, s_average
	else:
		return 3, u_average
'''

'''
h_test = l_img[0]
for y in range(len(l_img[0])):
	for x in range(len(l_img[0][0])):
		if l_img[0][y][x] == 0:
			h_test[y][x] = [0, 0, 0]
		else:
			h_test[y][x] = [255, 255, 255]

print(image_match(h_test))
'''
