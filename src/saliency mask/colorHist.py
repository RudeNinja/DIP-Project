def colorHist(img,colorLevels):
    #each color channel has to be divided into colorLevels number of bins:

    #Assume that we are working with a quantized image

    im = img.astype(int)

    chist = np.zeros((colorLevels+1,colorLevels+1,colorLevels+1))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            chist[im[i,j,0],im[i,j,1],im[i,j,2]] += 1

    nonzero_c = []
    nonzero_f = []
    for i in range(chist.shape[0]):
      for j in range(chist.shape[1]):
        for k in range(chist.shape[2]):
          if chist[i,j,k]!=0:
            nonzero_f.append(chist[i,j,k])
            nonzero_c.append((i,j,k))

    nf = nonzero_f
    zipped_lists = zip(nonzero_f, nonzero_c) 
    sorted_zipped_lists = sorted(zipped_lists,reverse=True) 
    nonzero_c = [element for _, element in sorted_zipped_lists]
    nonzero_f = nf.sort(reverse=True)

    
    return np.array(nonzero_c),np.array(nf);

