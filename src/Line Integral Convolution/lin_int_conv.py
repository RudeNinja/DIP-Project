def lic(noisy_image,vectorlines,radi_field = 25,kernelsize = 5):
    
    h,w = noisy_image.shape
    checked = np.zeros((h,w))
    output = np.zeros((h,w))

    filter = np.arange(1,kernelsize+1)
    filter = np.minimum(filter,filter[::-1])
    filter = filter/np.sum(filter)
    def func(y,t):
        return bilinear(vectorlines,y)

    coordinates = []    
    for i in range(h):
        for j in range(w):
            coordinates.append((i,j))

    random.shuffle(coordinates)
    for sp_pix in coordinates:
        i,j = sp_pix
        if checked[i][j] == 0:

            st_line = np.zeros((2*radi_field + 1,2))
            st_line[radi_field][0] = i 
            st_line[radi_field][1] = j 
            
            move_ahead = scipy.integrate.odeint(func,[i,j],np.arange(radi_field + 1),rtol=1e-2,atol=1e-2)
            move_back = scipy.integrate.odeint(func,[i,j],-np.arange(radi_field ),rtol=1e-2,atol=1e-2)

            st_line[radi_field:,:] = move_ahead
            st_line[:radi_field,:] = move_back[::-1]
            
            vals = []
            for k in range(2*radi_field + 1):
                vals.append(bilinear(noisy_image,st_line[k,:],False))

            vals = np.array(vals)
            conv_vals = scipy.signal.convolve(vals,filter,mode = 'same')
            ii_lin = np.floor(st_line[kernelsize :-(kernelsize ),0]).astype(int)
            jj_lin = np.floor(st_line[kernelsize :-(kernelsize ),1]).astype(int)

            conv_lin = conv_vals[kernelsize :-(kernelsize )]
            for i in range(len(ii_lin)):
                if ii_lin[i] >= 0 and ii_lin[i] < h and jj_lin[i] >= 0 and jj_lin[i] < w:
                    checked[ii_lin[i]][jj_lin[i]] += 1
                    output[ii_lin[i]][jj_lin[i]] += conv_lin[i]

    return (output/checked)/np.max(output/checked)
