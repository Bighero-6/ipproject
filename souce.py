import cv2
import numpy as np

digits = cv2.imread("digits.png",cv2.IMREAD_GRAYSCALE)
testdigits = cv2.imread("test_digits.png",cv2.IMREAD_GRAYSCALE)
cell = []
rows = np.vsplit(digits,50)

#identify individual cells
for row in rows:
    row_cell = np.hsplit(row,50)
    for cells in row_cell:
        cells = cells.flatten()#similar to ravel(nDarray -> 1D)
        cell.append(cells)
cell = np.array(cell, dtype = np.float32)
k = np.arange(10)
cell_labels = np.repeat(k,250)

test_digits = np.vsplit(testdigits,50)

test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells,dtype=np.float32)
#knn
knn = cv2.ml.KNearest_create() 
knn.train(cell,cv2.ml.ROW_SAMPLE,cell_labels)
ret , result , neigbours , distance = knn.findNearest(test_cells,k=1)

#print(result)
cv2.imshow("data",digits)
cv2.imshow("test",testdigits)
cv2.waitKey(0)
cv2.destroyAllWindows()
