def bilinear(img, coord, norm=True):
    
    h,w= img.shape[0],img.shape[1]
    a = int(np.floor(coord[0]))
    b = int(np.floor(coord[1]))
    alpha = coord[0]  - a
    beta = coord[1]  - b

    if a < 0 : a = 0
    elif a >= h: a = h - 1
    if b < 0 : b = 0
    elif b >= w: b = w - 1

    if a + 1 < 0 : da = 0
    elif a + 1 >= h: da = h - 1
    else: da = a + 1
    if b + 1 < 0 : db = 0
    elif b + 1 >= w: db = w - 1
    else: db = b + 1
    
    line1 = img[a, b] * (1 - beta) + img[a, db] * beta
    line2 = img[da, b] * (1 - beta) + img[da, db] * beta
    result = line1 * (1 - alpha) + line2 * alpha

    if norm:
        result /= np.sum(result ** 2) + 0.0001

    return result
