# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 19:04:05 2018

@author: Cristian Silva
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.linalg as ln
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import spectral.io.envi as envi
import time
import pickle
from sklearn import metrics
import rasterio
import fiona
import scipy
from rasterio.mask import mask
import os
#from scipy.ndimage.filters import uniform_filter
#from scipy.ndimage.measurements import variance
##import cv2
#import scipy 
#from osgeo import gdal
#from osgeo import gdal_array
#from osgeo import osr
#from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from skopt import BayesSearchCVom sklearn.neural_network import MLPClassifier   

def read_config_file(folder1):
    """
    This function reads the header (ENVI .hdr) when images are .Bin (RadarSAT Juanma)
    """
    #file = open("D:\\Juanma - Agrisar 2009\\Full_data\\V0\\All\\RS2_OK5763_PK77137_DK74934_FQ2_20090603_132228_HH_VV_HV_VH_SLC.data\\"+"T11_mst_03Jun2009.hdr","r")
    file = open(folder1+"\\config.txt","r")
    img_config=file.readlines() 
    print(img_config)
    Nrow=int(img_config[1]) # seville
    Ncol=int(img_config[4])
#    Nrow=int(img_config[2]) # agrisar
#    Ncol=int(img_config[3])
    header =  np.array([Nrow, Ncol])
    return(Nrow, Ncol,header)
    
# Function to open an image (T11,T22,T33)
def Open_C_diag_element(filename,header,datatype):   
    f = open(filename, 'rb')
    img = np.fromfile(f, dtype=datatype, sep="")
    img = img.reshape(header).astype(datatype)
    return(img)

# Function to open a COMPLEX image (T12,T23,T23)
def Open_C_element(filename1,filename2,header,datatype):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """
    fR = open(filename1, 'rb')
    imgR = np.fromfile(fR, dtype=datatype, sep="")
    imgR = imgR.reshape(header).astype(datatype)

    fI = open(filename2, 'rb') # image data
    imgI = np.fromfile(fI, dtype=datatype, sep="")
    imgIr = imgI.reshape(header).astype(datatype)
    imgI = imgIr*1j
    img12=imgR+imgI
    return(img12)
    
def read_Img_components_bin(basis,folder1,header,datatype):
    if basis=="P":
        print ("Opening Image from coherency matrix elements...")
        T11=Open_C_diag_element(folder1+"T11.bin",header,datatype)
        T22=Open_C_diag_element(folder1+"T22.bin",header,datatype)
        T33=Open_C_diag_element(folder1+"T33.bin",header,datatype)
        T12=Open_C_element     (folder1+"T12_real.bin",folder1+"T12_imag.bin",header,datatype)
        T13=Open_C_element     (folder1+"T13_real.bin",folder1+"T13_imag.bin",header,datatype)
        T23=Open_C_element     (folder1+"T23_real.bin",folder1+"T23_imag.bin",header,datatype)
        title=""
        #visRGB_from_T(T22, T33, T11, "RGB from T components "+title)
    elif basis=="L":
        print ("Opening Image C elements...")
        T11=Open_C_diag_element(folder1+"C11.bin",header,datatype)
        T22=Open_C_diag_element(folder1+"C22.bin",header,datatype)
        T33=Open_C_diag_element(folder1+"C33.bin",header,datatype)
        T12=Open_C_element     (folder1+"C12_real.bin",folder1+"C12_imag.bin",header,datatype)
        T13=Open_C_element     (folder1+"C13_real.bin",folder1+"C13_imag.bin",header,datatype)
        T23=Open_C_element     (folder1+"C23_real.bin",folder1+"C23_imag.bin",header,datatype)
        title=""
        #visRGB(T11, T22, T33, "RGB from C components "+title)
        return(T11,T22,T33,T12,T13,T23)
 
def C_to_T(C,U):
    Uinv=U.getH()
    T=U.dot(C).dot(Uinv)
    return (T) 

def array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype):
    """    
    inputs: 
    type_of_matrix: whether components are in lexicographic (L) or pauli (P) basis 
    folder: path to find the components
    in_format: Format of the components: .Bin like Juanma (Bin) or .img (img) got after preprocessing with SNAP
    header: Obtained after reading the config_file
    datatype:
    """
    # The components will be called C even if they are in Pauli basis to avoid creating other set of variables
    if in_format=="Bin":
        C11,C22,C33,C12,C13,C23=read_Img_components_bin(basis,folder,header,datatype)
    elif in_format=="img":
        C11,C22,C33,C12,C13,C23=read_Img_components_SNAP(basis,folder,header,datatype)
    print(C11.shape)    
    C21=np.conj(C12)
    C31=np.conj(C13)
    C32=np.conj(C23)
    ####################################### crop it as the ROI ######################
    x_min=ROI_size[0]
    x_max=ROI_size[1]
    y_min=ROI_size[2]
    y_max=ROI_size[3]
#    print(y_min)
    print("ymax="+str(y_max))
#    print(x_min)
    print("xmax="+str(x_max))
    # Compute image size
    da=(x_max-x_min)
    dr=(y_max-y_min) 
        
    C11=C11[y_min:y_max,x_min:x_max] # 
    C22=C22[y_min:y_max,x_min:x_max]
    C33=C33[y_min:y_max,x_min:x_max] 
    C12=C12[y_min:y_max,x_min:x_max]
    C13=C13[y_min:y_max,x_min:x_max]
    C23=C23[y_min:y_max,x_min:x_max]
    C21=C21[y_min:y_max,x_min:x_max]
    C31=C31[y_min:y_max,x_min:x_max]
    C32=C32[y_min:y_max,x_min:x_max]
    print(C11.shape)
    ###################################### create covariance matrix for each pixel #####################
    C=np.zeros([dr,da,3,3],dtype=np.complex64)
    C[:,:,0,0]=C11
    C[:,:,0,1]=C12
    C[:,:,0,2]=C13
    C[:,:,1,0]=C21
    C[:,:,1,1]=C22
    C[:,:,1,2]=C23
    C[:,:,2,0]=C31
    C[:,:,2,1]=C32
    C[:,:,2,2]=C33
    # if the components are in lexicographic basis, transform to pauli 
    if basis == "L":
        print("C to T change of basis ...")
        # reshape so that the function C_to_T can be applied. This could be further improved ...
        # for every position of array apply the function C_to_T
        C1 = C.reshape(np.array([dr*da,3,3])).astype('complex64')
        C=np.zeros([dr,da,3,3],dtype=np.complex64)
        U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
        U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
        All_T_matrices3=np.asarray([C_to_T(item,U) for item in C1])  # item = pixel
        del C1
        #U.dot(C).dot(Uinv)
        C[:,:,:,:]=All_T_matrices3.reshape(np.array([dr, da,3,3])).astype('complex64') # reshape as the image  
        del All_T_matrices3
    return(C)

def read_Img_components_from_stack_SNAP(basis,folder1,a):
    """ This function opens the elements of the coherency matrices obtained from SNAP by
    radar - polarimetric - Polarimetric matrix generation """
    if basis=="P":
        lib = envi.open(folder1 +"\\"+a[0].split(".")[0] + '.hdr')
        Ncol=lib.ncols
        Nrow=lib.nrows
        header =  np.array([Nrow, Ncol])
        datatype = lib.dtype
        T11=Open_C_diag_element(folder1+"\\"+a[0],header,datatype)
        T22=Open_C_diag_element(folder1+"\\"+a[5],header,datatype)
        T33=Open_C_diag_element(folder1+"\\"+a[-1],header,datatype)
        T12=Open_C_element     (folder1+"\\"+a[2],folder1+"\\"+a[1],header,datatype)
        T13=Open_C_element     (folder1+"\\"+a[4],folder1+"\\"+a[3],header,datatype)
        T23=Open_C_element     (folder1+"\\"+a[7],folder1+"\\"+a[6],header,datatype)
    elif basis=="L":
        lib = envi.open(folder1 +"C11" + '.hdr')
        Ncol=lib.ncols
        Nrow=lib.nrows
        header =  np.array([Nrow, Ncol])
        datatype = lib.dtype
        print ("Opening Image from coherency matrix elements...")
        T11=Open_C_diag_element(folder1+"C11.img",header,datatype)
        T22=Open_C_diag_element(folder1+"C22.img",header,datatype)
        T33=Open_C_diag_element(folder1+"C33.img",header,datatype)
        T12=Open_C_element     (folder1+"C12_real.img",folder1+"C12_imag.img",header,datatype)
        T13=Open_C_element     (folder1+"C13_real.img",folder1+"C13_imag.img",header,datatype)
        T23=Open_C_element     (folder1+"C23_real.img",folder1+"C23_imag.img",header,datatype)            
    return(T11,T22,T33,T12,T13,T23)  

def array2D_of_coherency_matrices_from_stack_SNAP(folder,basis,in_format,ROI_size,header,datatype,a):
    """    
    inputs: 
    type_of_matrix: whether components are in lexicographic (L) or pauli (P) basis 
    folder: path to find the components
    in_format: Format of the components: .Bin like Juanma (Bin) or .img (img) got after preprocessing with SNAP
    header: Obtained after reading the config_file
    datatype:
    """
    # The components will be called C even if they are in Pauli basis to avoid creating other set of variables
    if in_format=="Bin":
        C11,C22,C33,C12,C13,C23=read_Img_components_bin(basis,folder,header,datatype)
    elif in_format=="img":
        C11,C22,C33,C12,C13,C23=read_Img_components_from_stack_SNAP(basis,folder,a)
    print(C11.shape)    
    C21=np.conj(C12)
    C31=np.conj(C13)
    C32=np.conj(C23)
    ####################################### crop it as the ROI ######################
    x_min=ROI_size[0]
    x_max=ROI_size[1]
    y_min=ROI_size[2]
    y_max=ROI_size[3]
#    print(y_min)
    print("ymax="+str(y_max))
#    print(x_min)
    print("xmax="+str(x_max))
    # Compute image size
    da=(x_max-x_min)
    dr=(y_max-y_min) 
        
    C11=C11[y_min:y_max,x_min:x_max] # 
    C22=C22[y_min:y_max,x_min:x_max]
    C33=C33[y_min:y_max,x_min:x_max] 
    C12=C12[y_min:y_max,x_min:x_max]
    C13=C13[y_min:y_max,x_min:x_max]
    C23=C23[y_min:y_max,x_min:x_max]
    C21=C21[y_min:y_max,x_min:x_max]
    C31=C31[y_min:y_max,x_min:x_max]
    C32=C32[y_min:y_max,x_min:x_max]
    print(C11.shape)
    ###################################### create covariance matrix for each pixel #####################
    C=np.zeros([dr,da,3,3],dtype=np.complex64)
    C[:,:,0,0]=C11
    C[:,:,0,1]=C12
    C[:,:,0,2]=C13
    C[:,:,1,0]=C21
    C[:,:,1,1]=C22
    C[:,:,1,2]=C23
    C[:,:,2,0]=C31
    C[:,:,2,1]=C32
    C[:,:,2,2]=C33
    # if the components are in lexicographic basis, transform to pauli 
    if basis == "L":
        print("C to T change of basis ...")
        # reshape so that the function C_to_T can be applied. This could be further improved ...
        # for every position of array apply the function C_to_T
        C1 = C.reshape(np.array([dr*da,3,3])).astype('complex64')
        C=np.zeros([dr,da,3,3],dtype=np.complex64)
        U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
        U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
        All_T_matrices3=np.asarray([C_to_T(item,U) for item in C1])  # item = pixel
        del C1
        #U.dot(C).dot(Uinv)
        C[:,:,:,:]=All_T_matrices3.reshape(np.array([dr, da,3,3])).astype('complex64') # reshape as the image  
        del All_T_matrices3
    return(C)   
    
def visRGB(img1, img2, img3, title):
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
    R=(img1-img3)
    G=img2
    B=(img1+img3)
    iRGB[:,:,0] = np.abs(R)/(np.abs(R).nanmean()*5)
    iRGB[:,:,1] = np.abs(G)/(np.abs(G).nanmean()*5)
    iRGB[:,:,2] = np.abs(B)/(np.abs(B).nanmean()*5)
    iRGB[np.abs(iRGB) > 1] = 1
#    
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()
    return(iRGB) 
    
def visRGB_from_T(img1, img2, img3,
           title,
           scale1 = [],
           scale2 = [],
           scale3 = [],factor=3,save=0,outall=""):
    """
    Visualise the RGB of a single acquisition
    """           
    if scale1 == []:
       scale1 = (0, np.abs(img1).mean()*2)
    if scale2 == []:
       scale2 = (0, np.abs(img2).mean()*2)
    if scale3 == []:
       scale3 = (0, np.abs(img3).mean()*2)
    size = np.shape(img1)           
    iRGB = np.zeros([size[0],size[1],3])
    iRGB[:,:,0] = np.abs(img1)/(np.abs(np.nanmean(img1)*factor))
    iRGB[:,:,1] = np.abs(img2)/(np.abs(np.nanmean(img2)*factor))
    iRGB[:,:,2] = np.abs(img3)/(np.abs(np.nanmean(img3)*factor))
    #iRGB[:,:,2] = np.abs(img3)/(np.abs(img3).mean()*2.5)
    #iRGB[np.abs(iRGB) > 1] = 1
    iRGB=np.nan_to_num(iRGB)        
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    # For multitemporal filter and additive change matrices, multiply by 7. For multiplicative change matrices divide by 7
    ax1.imshow(np.abs(iRGB)) 
    plt.axis("off")
    plt.tight_layout()
    if save == 1:
        print("saving "+title+" ..." )
        outall1= outall+'\\'+title+'.png'
        fig.savefig(outall1, bbox_inches='tight',dpi=1200)    
    return 

def eigendecomposition_vectorized(T):
    print("Covariance matrices eigendecomposition ...")
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    """
    w(…, M) ndarray
    The eigenvalues in ascending order, each repeated according to its multiplicity.
    v{(…, M, M) ndarray, (…, M, M) matrix}
    The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]. Will return a matrix object if a is a matrix object.
    """
    #ind = np.argsort(eigenvalues)# organize eigenvalues from higher to lower value
    eigenvalues[eigenvalues < 0] = 0 
    
    #L1 = eigenvalues[ind[2]] # already come sorted from np.linalg.eigh
    #L2 = eigenvalues[ind[1]]
    #L3 = eigenvalues[ind[0]]
    
    L1 = eigenvalues[:,:,2]
    L2 = eigenvalues[:,:,1]
    L3 = eigenvalues[:,:,0]
    
    U_11=np.abs(eigenvectors[:,:,0,2]) 
    U_21=np.abs(eigenvectors[:,:,1,2])
    U_31=np.abs(eigenvectors[:,:,2,2])
     
    U_12=np.abs(eigenvectors[:,:,0,1])
    U_22=np.abs(eigenvectors[:,:,1,1])
    U_32=np.abs(eigenvectors[:,:,2,1])
    
    U_13=np.abs(eigenvectors[:,:,0,0])
    U_23=np.abs(eigenvectors[:,:,1,0])
    U_33=np.abs(eigenvectors[:,:,2,0])     
    return(L1,L2,L3,  U_11,U_21,U_31,  U_12,U_22,U_32,  U_13,U_23,U_33)


def Alpha_Entropy_Anisotropy_decomp(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title1,save,outall,factor,plot="yes"):
    print("Plotting alpha/Entropy/Anisotropy results  ...")
    P1=(L1/(L1+L2+L3))
    entropy1=((-P1)*(np.log(P1)/np.log(3)))
    P2=(L2/(L1+L2+L3))
    entropy2=((-P2)*(np.log(P2)/np.log(3)))
    P3=(L3/(L1+L2+L3))
    entropy3=((-P3)*(np.log(P3)/np.log(3)))
            
    Entropy=entropy1+entropy2+entropy3
    Anisotropy=((L2-L3)/(L2+L3))
    
    alpha1=np.arccos(U_11)
    alpha2=np.arccos(U_12)
    alpha3=np.arccos(U_13)
    
    beta1=np.arccos((U_21)/np.sin(alpha1))
    beta2=np.arccos((U_22)/np.sin(alpha2))
    beta3=np.arccos((U_23)/np.sin(alpha3))
    
    L_avg=(P1*L1)+(P2*L2)+(P3*L3)
    alpha_avg=(P1*alpha1)+(P2*alpha2)+(P3*alpha3)
    beta_avg =(P1*beta1) +(P2*beta2) +(P3*beta3)
    
    L_avg = np.nan_to_num(L_avg)
    alpha_avg = np.nan_to_num(alpha_avg)
    beta_avg = np.nan_to_num(beta_avg)
    
    vmin=0
    vmax=np.pi/2
    title=title1+" Dominant Alpha angle - rad"
    alpha1_avg=plot_descriptor(alpha1,vmin,vmax,title)
    
    #title=title1+" Alpha 2 - rad"
    #alpha2_avg=plot_descriptor(alpha2,vmin,vmax,title)
    #
    #title=title1+" Alpha 3 - rad"
    #alpha3_avg=plot_descriptor(alpha3,vmin,vmax,title)
    
#    title=title1+" Average Alpha angle - Rad" 
#    alpha_avg=plot_descriptor(alpha_avg,vmin,vmax,title)
    
#    title=title1+" Dominant Beta angle"
#    beta1_avg=plot_descriptor(beta1,vmin,vmax,title) 
#    
#    title=title1+" Average Beta angle - Rad"
#    beta_avg=plot_descriptor(beta_avg,vmin,vmax,title)
    
    vmax=1
    title=title1+" Entropy"
    Entropy_avg=plot_descriptor(Entropy,vmin,vmax,title)
    
    title=title1+" Anisotropy"
    Anisotropy_avg=plot_descriptor(Anisotropy,vmin,vmax,title)
    
#    title=title1+" Lamda 1"
#    L1_avg=plot_descriptor(L1,vmin,vmax,title)
    
    #title=title1+" Lamda 2"
    #L2_avg=plot_descriptor(L2,vmin,vmax,title)
    #
    #title=title1+" Lamda 3"
    #L3_avg=plot_descriptor(L3,vmin,vmax,title)

    title=title1+" Main scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
    R_avg,G_avg,B_avg=scattering_mechanism(L_avg,alpha_avg,beta_avg,title,save,outall,factor)
    
    if plot=="No":
        plt.close('all')
        
    #R=(np.sqrt(np.abs(L_avg)))*(np.sin(alpha_avg))*(np.sin(beta_avg)) 
    return(alpha_avg,Entropy,Anisotropy,R_avg,G_avg,B_avg)

def plot_descriptor(descriptor,vmin,vmax,title):
    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax10.set_title(title)
    im10=ax10.imshow(descriptor, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax10.axis('off')
    color_bar(im10) # 
    descriptor = np.nan_to_num(descriptor)  # nan to zero
    descriptor[descriptor == (np.pi/2)] = 0 # 
    descriptor_avg=(np.sum(descriptor))/np.count_nonzero(descriptor)
    descriptor = np.nan_to_num(descriptor)
    plt.tight_layout()   
    return (descriptor_avg)

def color_bar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    #ax.tick_params(labelsize=10)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)   
    return fig.colorbar(mappable, cax=cax,orientation="horizontal")    

def scattering_mechanism(L,alpha,beta,title,save,outall,factor):
    # As in the PolSARpro tutorial: https://earth.esa.int/documents/653194/656796/Polarimetric_Decompositions.pdf   
    R=(np.sqrt(np.abs(L)))*(np.sin(alpha))*(np.cos(beta)) 
    G=(np.sqrt(np.abs(L)))*(np.sin(alpha))*(np.sin(beta)) 
    B=(np.sqrt(np.abs(L)))*(np.cos(alpha))                
    visRGB_from_T(R, G, B,title,factor=factor)
    return(R,G,B)

def full_Alpha_Entropy_Anisotropy_decomp(folder,basis,in_format,ROI_size,header,datatype,title1,save,outall):
    T=array2D_of_coherency_matrices(folder,basis,in_format,ROI_size,header,datatype)
    visRGB_from_T(T[:,:,1,1], T[:,:,2,2], T[:,:,0,0],"",factor=1.5)
    L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33=eigendecomposition_vectorized(T)
    alpha1_avg,Entropy_avg,Anisotropy_avg,R_avg,G_avg,B_avg=Alpha_Entropy_Anisotropy_decomp(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title1,save,outall,factor=3,plot="yes")

def bi_date_Quad_pol_CD(T11,T22):
    """
    ####################################### Tc Increase ############################
    """
    print("Processing added scattering mechanisms...")
    Tc = T22 - T11
    L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33=eigendecomposition_vectorized(Tc)

    title1="Increase in the"
    save=""
    outall=""
    alpha1_avg,Entropy_avg,Anisotropy_avg,R_avg,G_avg,B_avg=Alpha_Entropy_Anisotropy_decomp(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title1,save,outall,factor=3,plot="No")
    
    """
    ####################################### Tc decrease ############################
    """
    print("Processing removed scattering mechanisms...")
    Tc = T11 - T22
    
    """  
    ###########################################################################################################################################################
    # return the eigenvalue and eigenvector decomposition. 2D array of image size, in which each position contains 3 eigenvelues, or 3x3 eigenvectors #########
    """
    L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33=eigendecomposition_vectorized(Tc)
    
    """  
    ###########################################################################################################################################################
    ###########################################################################################################################################################
    # return the alpha, entropy, anisotropy and main scattering mechanishms images (Cloude-Pottier) #########
    ###########################################################################################################################################################
    ###########################################################################################################################################################
    """
    title1="Decrease in the"
    save=""
    outall=""
    alpha1_avg,Entropy_avg,Anisotropy_avg,R_avg_dec,G_avg_dec,B_avg_dec=Alpha_Entropy_Anisotropy_decomp(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title1,save,outall,factor=3,plot="No")
    
    return(R_avg,G_avg,B_avg,R_avg_dec,G_avg_dec,B_avg_dec)
    
#    #RGB[np.abs(RGB) > 1] = 1    
#    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
#    ax1.set_title(title)
#    ax1.imshow(np.abs(RGB))
#    plt.axis("off")
#    plt.tight_layout() 
#    if save ==1:
#        print("saving "+title+" ..." )
#        outall1= outall+title+'.png'
#        fig.savefig(outall1, bbox_inches='tight')   
#    plt.show()
    
def Plot_change_matrix(Matrix,title,save,outall):
    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #a = np.expand_dims(a, axis=0)  # or axis=1
    ax10.set_title(title)
    ax10.imshow(Matrix)
    ax10.axis('off')
    plt.tight_layout() 
    #color_bar(im10)
    if save ==1:
        print("saving "+title+" ..." )
        outall1= outall+title+'.png'
        fig.savefig(outall1, bbox_inches='tight')   
    
    plt.show()
    return

def Plot_all_change_matrices(change_matrix,title1,save,outall,factor): 
    title="No_Scaling_"+title1
    Plot_change_matrix(change_matrix*factor,title,save,outall)
    
#    change_matrix1=change_matrix*0.25
#    title=title1+" lineal scale"
#    Plot_change_matrix(change_matrix1,title,save,outall)
    change_matrix = np.nan_to_num(change_matrix)
    change_matrix_=np.zeros([change_matrix.shape[0],change_matrix.shape[1],3])
    change_matrix_[:,:,0]=    change_matrix[:,:,0]/((np.abs(change_matrix[:,:,0]).mean()*factor))
    change_matrix_[:,:,1]=    change_matrix[:,:,1]/((np.abs(change_matrix[:,:,1]).mean()*factor))
    change_matrix_[:,:,2]=    change_matrix[:,:,2]/((np.abs(change_matrix[:,:,2]).mean()*factor))
    title=title1
    Plot_change_matrix(change_matrix_,title,save,outall)
    return

def interpolate_matrix(Source, days):
    """
    Source: Each channel of the change matrix, i.e. 5x5 matrix if there are 5 dates
    days: Number of days to interpolate
    We first interpolate the 1st dimension and using this result we interpolate in the second dimension
    """
#    Source=np.array([ [0, 2, 5, 1, 0.5],
#                      [1, 0, 0, 2, 3],
#                      [3, 3, 0, 8, 6],
#                      [2, 4, 6, 0, 3],
#                      [4, 5, 7, 1, 0]])
    #x = np.arange(0, 15)
    #days=30
    
    x = np.arange(0, Source.shape[0])
    fit = scipy.interpolate.interp1d(x, Source, axis=0)
    Target = fit(np.linspace(0, Source.shape[0]-1, days))
    #Target = fit(np.linspace(0, 15-1, 15))
    #print(Target)
    
    x = np.arange(0,Target.shape[1])
    #x = np.arange(0, 15)
    fit = scipy.interpolate.interp1d(x, Target, axis=1)
    Target = fit(np.linspace(0, Target.shape[1]-1, days))
    print(Target)
    
    return(Target)    

def read_raster_and_shp(tiff_path,tiff_name,shp_path,shp_name):
    ################################################# Read tif raster of the SAR image ##############################
    # Saved after doing the preprocessing. This contains geo info to map from 
    # geocoordinates to image coordinates
    raster = rasterio.open(tiff_path+tiff_name)
    ################################################# read shp
    # Read shapefile with fiona package
    Polygons_in_shp=fiona.open(shp_path+shp_name, "r")
    return(raster,Polygons_in_shp)

def crop_MT_datacube(poly,raster,Polygons_in_shp,T_mult):
    """
    This function crops a multitemporal datacube to the size of a given polygon
    """
    ################################################# Crop the polygon
    for feature in Polygons_in_shp:
        if feature['properties']['IDENT']== poly:
            #print(feature['properties']['IDENT'])
            polygon_A=feature["geometry"] # filter the polygon from shp
    ################################################# Mask the polygon
    # use rasterio to mask and obtain the pixels inside the polygon. 
    # Because the mask method of rasterio requires an interable object to work, we do:
    polygon_A_list=[polygon_A] 
    # out_image in line below is a np array of the polygon desired
    out_image, out_transform = mask(raster, polygon_A_list, crop=True) 
    ################################################ Get image coord of the corners of the polygon(subset)
    numpy_polygon=out_image[:,:,:][0]
    # affine transform of rasterio 
    subset_origin=~raster.transform * (out_transform[2], out_transform[5])
    # coordinates of top-left 
    subset_originX = int(subset_origin[0])
    subset_originY = int(subset_origin[1])
    # size of the cropped subset after masking
    pixels_in_row=int(numpy_polygon.shape[1])
    pixels_in_col=int(numpy_polygon.shape[0])
    
    # Crop the image to some known size otherwise comment (disable) the lines below
    # Polygon
    x_min=subset_originY
    x_max=subset_originY+pixels_in_col
    y_min=subset_originX
    y_max=subset_originX+pixels_in_row

    T_small=T_mult[:,x_min:x_max,y_min:y_max,:,:]
    return(T_small,numpy_polygon)

    
def array_to_raster_rasterIO(np_array,newRaster,rows,cols,originX,originY,crss,pixelWidth, pixelHeight):
    # Obtain an Affine transformation for a georeferenced raster given the coordinates 
    # of its upper left corner west, north and pixel sizes xsize, ysize.
    from rasterio.transform import from_origin
    transform = from_origin(originX, originY, pixelWidth, pixelHeight)
    
    new_dataset = rasterio.open(newRaster, 'w', driver='GTiff',
                                height = rows, width = cols,
                                count=1, dtype=str(np_array.dtype),                            
                                crs=crss,
                                transform=transform)
    
    new_dataset.write(np_array, 1)
    new_dataset.close()


# this function use GDAL but I had problems when opening with QGIS and also
# because it wasnt recognizing the EPSG. Use the rasterio option instead 
#def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,EPSG):
#    """
#    Function to save a numpy raster as a geotiff when only the origin and pixel size are known
#    Inputs: 
#        newRasterfn: Path and name where to save the raster
#        rasterOrigin: Tuple with coordinates. E.g. (-123.25745,45.43013)
#        pixelWidth: E.g. 10
#        pixelHeight: E.g. 10
#        array: 2D numpy array with pixel values
#    Outputs:
#        Geotiff raster
#    """
#    cols = array.shape[1]
#    rows = array.shape[0]
#    originX = rasterOrigin[0]
#    originY = rasterOrigin[1]
#
#    driver = gdal.GetDriverByName('GTiff')
#    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
#    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
#    #(upper_left_x, x_resolution, x_skew, upper_left_y, y_skew, y_resolution)
#    outband = outRaster.GetRasterBand(1)
#        #normalize the array to a 0-255 scale
##    if np.min(array) < 0:
##        array = array + np.abs(np.min(array))
##    #Normalize array to a 0-255 scale for raster channel
##    mag_grid2 = ((array - np.min(array))/(np.max(array) - np.min(array)))*256
#    
#    #outband.WriteArray(mag_grid2)
#    outband.WriteArray(array)
#    outRasterSRS = osr.SpatialReference()
#    outRasterSRS.ImportFromEPSG(EPSG)
#    #4326
#    #102579 UTM29N
#    outRaster.SetProjection(outRasterSRS.ExportToWkt())
#    outband.FlushCache()
#
#def array_to_raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,EPSG):
#    import rasterio
#    from rasterio.transform import from_origin
#    cols = array.shape[1]
#    rows = array.shape[0]
#    originX = rasterOrigin[0]
#    originY = rasterOrigin[1]
#    
#    # Obtain an Affine transformation for a georeferenced raster given the coordinates 
#    # of its upper left corner west, north and pixel sizes xsize, ysize.
#    transform = from_origin(originX, originY, pixelWidth, pixelHeight)
#    
#    new_dataset = rasterio.open(newRasterfn, 'w', driver='GTiff',
#                                height = rows, width = cols,
#                                count=1, dtype=str(array.dtype),
#                                crs='EPSG:'+str(EPSG),
#                                #crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
#                                transform=transform)
#    
#    new_dataset.write(array, 1)
#    new_dataset.close()

def read_config_file_snap(folder1):
    """
    This function reads the header (ENVI .hdr) generated by SNAP
    """
    lib = envi.open(folder1 + '.hdr')
#    header =  np.array([lib.nrows, lib.ncols])
    Ncol=lib.ncols
    Nrow=lib.nrows
    #header =  np.array([Nrow, Ncol])
    #datatype = lib.dtype
    header =  np.array([Nrow, Ncol])
    return(Nrow, Ncol, header)
    
def read_Img_components_SNAP(basis,folder1,header,datatype):
    """ This function opens the elements of the coherency matrices obtained from SNAP by
    radar - polarimetric - Polarimetric matrix generation """
    if basis=="P":
        lib = envi.open(folder1 +"T11" + '.hdr')
        Ncol=lib.ncols
        Nrow=lib.nrows
        header =  np.array([Nrow, Ncol])
        datatype = lib.dtype
        T11=Open_C_diag_element(folder1+"T11.img",header,datatype)
        T22=Open_C_diag_element(folder1+"T22.img",header,datatype)
        T33=Open_C_diag_element(folder1+"T33.img",header,datatype)
        T12=Open_C_element     (folder1+"T12_real.img",folder1+"T12_imag.img",header,datatype)
        T13=Open_C_element     (folder1+"T13_real.img",folder1+"T13_imag.img",header,datatype)
        T23=Open_C_element     (folder1+"T23_real.img",folder1+"T23_imag.img",header,datatype)
    elif basis=="L":
        lib = envi.open(folder1 +"C11" + '.hdr')
        Ncol=lib.ncols
        Nrow=lib.nrows
        header =  np.array([Nrow, Ncol])
        datatype = lib.dtype
        print ("Opening Image from coherency matrix elements...")
        T11=Open_C_diag_element(folder1+"C11.img",header,datatype)
        T22=Open_C_diag_element(folder1+"C22.img",header,datatype)
        T33=Open_C_diag_element(folder1+"C33.img",header,datatype)
        T12=Open_C_element     (folder1+"C12_real.img",folder1+"C12_imag.img",header,datatype)
        T13=Open_C_element     (folder1+"C13_real.img",folder1+"C13_imag.img",header,datatype)
        T23=Open_C_element     (folder1+"C23_real.img",folder1+"C23_imag.img",header,datatype)            
    return(T11,T22,T33,T12,T13,T23)    

def Multitemporal_filter(C):
    C_avg=np.mean(C,axis=(0))
    visRGB_from_T(C_avg[:,:,1,1], C_avg[:,:,2,2], C_avg[:,:,0,0],"")
    return(C_avg)
    
def plot_backscatter_log_scale(img,title,is_in="log"):
    if is_in=="linear":
        img=10*np.log10(img)
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(img), cmap = 'gray', vmin=0, vmax=np.abs(img).mean()*2)
    plt.axis("off")
    plt.tight_layout()
    #fig.colorbar(im1,ax=ax1)
        


def plot_descriptor_in_ROI(descriptor,vmin,vmax,title,x,y):
    mask_only_ROI(descriptor,x,y)
    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax10.set_title(title)
    im10=ax10.imshow(descriptor, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax10.axis('off')
    color_bar(im10)
    plt.tight_layout()
    return

def Tmatrix(hh, hv, vv, win):
    """
    This routine generates the elements of the Coherency matrix, given the 
    images in lexicographic basis and the size of the moving window
    """
    # Elements of the Pauli basis
    p1 = hh+vv
    p2 = hh-vv
    p3 = 2*hv
    
#    kernel for averaging
    kernel  = np.ones((win,win),np.float32)/(np.power(win,2))
    
    Taa = signal.convolve2d(np.power(np.abs(p1),2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tbb = signal.convolve2d(np.power(np.abs(p2),2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tcc = signal.convolve2d(np.power(np.abs(p3),2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tab = signal.convolve2d(p1*np.conj(p2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tac = signal.convolve2d(p1*np.conj(p3), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tbc = signal.convolve2d(p2*np.conj(p3), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
                            
    return(Taa, Tbb, Tcc, Tab, Tac, Tbc)
    
def Lee_Filter(img,win):
    """
    This module aplies Lee Filter with a win x win window 
    """    
    img_mean = uniform_filter(img, (win, win))
    img_sqr_mean = uniform_filter(img**2, (win, win))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance =variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return(img_output)
    






        
def Alp_Ent_Ani_analysis_SMs(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title1):
    print("Plotting scattering mechanisms provided by the eigenvector-eigenvalue based decomposition  ...")
    P1=(L1/(L1+L2+L3))
    P2=(L2/(L1+L2+L3))
    P3=(L3/(L1+L2+L3))
            
    alpha1=np.arccos(np.abs(U_11))
    alpha2=np.arccos(np.abs(U_12))
    alpha3=np.arccos(np.abs(U_13))
    
    beta1=np.arccos(np.abs(U_21)/np.sin(alpha1))
    beta2=np.arccos(np.abs(U_22)/np.sin(alpha2))
    beta3=np.arccos(np.abs(U_23)/np.sin(alpha3))
    
    L_avg=(P1*L1)+(P2*L2)+(P3*L3)
    alpha_avg=(P1*alpha1)+(P2*alpha2)+(P3*alpha3)
    beta_avg=(P1*beta1)+(P2*beta2)+(P3*beta3)
    L_avg = np.nan_to_num(L_avg)
    alpha_avg = np.nan_to_num(alpha_avg)
    beta_avg = np.nan_to_num(beta_avg)
    
    title=title1+" Dominant scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
    R1_avg,G1_avg,B1_avg=scattering_mechanism(L1,alpha1,beta1,title)
    #
    title=title1+" Second scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
    R2_avg,G2_avg,B2_avg=scattering_mechanism(L2,alpha2,beta2,title)
    
    title=title1+" Third scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
    R3_avg,G3_avg,B3_avg=scattering_mechanism(L3,alpha3,beta3,title)
    
    title=title1+" Main scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
    R_avg,G_avg,B_avg=scattering_mechanism(L_avg,alpha_avg,beta_avg,title)
    
    return(R1_avg,G1_avg,B1_avg,R2_avg,G2_avg,B2_avg,R3_avg,G3_avg,B3_avg,R_avg,G_avg,B_avg)
    
def visRGB_L_Contrast(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33,title):
    """
    Visualise the RGB of a single acquisition
    """           
    size = np.shape(U_11)           
    iRGB1 = np.zeros([size[0],size[1],3])
    iRGB2 = np.zeros([size[0],size[1],3])
    iRGB3 = np.zeros([size[0],size[1],3])

    iRGB1[:,:,0] = U_21*L1
    iRGB1[:,:,1] = U_31*L1
    iRGB1[:,:,2] = U_11*L1
    #iRGB1[np.abs(iRGB1) > 1] = 1
    iRGB2[:,:,0] = U_22*L2
    iRGB2[:,:,1] = U_32*L2
    iRGB2[:,:,2] = U_12*L2  
    #iRGB2[np.abs(iRGB2) > 1] = 1                                          
    iRGB3[:,:,0] = U_23*L3
    iRGB3[:,:,1] = U_33*L3
    iRGB3[:,:,2] = U_13*L3  
    #iRGB3[np.abs(iRGB3) > 1] = 1
    iRGB=iRGB1+iRGB2+iRGB3
    #K=0.25
    #iRGB[np.abs(iRGB) > 1] = 1
    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()  
    iRGB = np.nan_to_num(iRGB)
    R_avg=np.sum(iRGB[:,:,0])/np.count_nonzero(iRGB[:,:,0])
    G_avg=np.sum(iRGB[:,:,1])/np.count_nonzero(iRGB[:,:,1])
    B_avg=np.sum(iRGB[:,:,2])/np.count_nonzero(iRGB[:,:,2])
    return(R_avg,G_avg,B_avg)


    
#for i in range(CM.shape[0]):
#    CM1[i,i,0]=0
#    CM1[i,i,1]=0
#    CM1[i,i,2]=0
#    
#from scipy import interpolate
#x = np.arange(0, CM.shape[0], 1)
#y = np.arange(0, CM.shape[0], 1)
#xx, yy = np.meshgrid(x, y)
#z = CM1[:,:,0]
#f0 = interpolate.interp2d(x, y, z, kind='linear')
#z = CM1[:,:,1]
#f1 = interpolate.interp2d(x, y, z, kind='linear')
#z = CM1[:,:,2]
#f2 = interpolate.interp2d(x, y, z, kind='linear')
#
#xnew = np.arange(0, CM.shape[0], 0.1)
#ynew = np.arange(0, CM.shape[0], 0.1)
#znew0 = f0(xnew, ynew)
#znew1 = f1(xnew, ynew)
#znew2 = f2(xnew, ynew)
#
#CM_INTE = np.zeros([xnew.shape[0],xnew.shape[0],3])
#CM_INTE[:,:,0] = znew0
#CM_INTE[:,:,1] = znew1
#CM_INTE[:,:,2] = znew2
    
def wishart(Zij,Zk):
    Zk_inv=np.linalg.inv(Zk)
    distance=np.linalg.norm(np.log(Zk)+np.trace(np.dot(Zk_inv,Zij)))
    #-10*math.log(np.linalg.norm(ZA1-ZA2)**2)
    #distance=np.linalg.norm(ZA1-ZA2)**2
    return(distance)

def geo_distance(Za1,Zb):
    Za_sqrt1=np.sqrt(Za1)
    Za_sqrt_inv1=ln.inv(Za_sqrt1)
    dgc1=Za_sqrt_inv1.dot(Zb).dot(Za_sqrt_inv1)
    dgc1=ln.logm(dgc1, disp=True)
    dgc_1=np.linalg.norm(dgc1)
    return(dgc_1)
    
def frob_dist(ZA1,ZA2):
    distance=-10*math.log10(np.linalg.norm(ZA1-ZA2)**2)
    #distance=np.linalg.norm(ZA1-ZA2)**2
    return(distance)

def rev_wishart(Zij,Zk):
    Zk_inv=np.linalg.inv(Zk)
    Zij_inv=np.linalg.inv(Zij)
    distance=np.linalg.norm(np.trace(np.dot(Zk,Zij_inv))+np.trace(np.dot(Zk_inv,Zij)))
    #-10*math.log(np.linalg.norm(ZA1-ZA2)**2)
    #distance=np.linalg.norm(ZA1-ZA2)**2
    return(distance)
    
def rev_wishart_images(C,C_I2,A,B):
    #Take as input the an array with shape [da,dr] where each position correspond to the 3x3 or 2x2 covariance matrices of a pixel 
    C_inv=np.linalg.inv(C)
    C_I2_inv=np.linalg.inv(C_I2)
    CC_I2_inv=np.matmul(C,C_I2_inv)
    C_invC_I2=np.matmul(C_inv,C_I2)
    
    print("Wishart distance image "+str(A+1)+" to class "+ str(B+1)+ "...")
    # First term of the equation (trace 1)
    tr1=CC_I2_inv[:,:,0,0]+CC_I2_inv[:,:,1,1]+CC_I2_inv[:,:,2,2]
     # Second term of the equation (trace 2)
    tr2=C_invC_I2[:,:,0,0]+C_invC_I2[:,:,1,1]+C_invC_I2[:,:,2,2]
    # Wishart similarity measure
    D=np.abs(tr1+tr2)
    #Returns array of size equal to the image size (da,dr)
    return(D)    
"""
From here are not really used 
"""
    
    
    
def eigendecomposition(Taa1,Tab1,Tac1,Tbb1,Tbc1,Tcc1,title1):
    """ Pixelwise: Slow and not used anymore """
    print("Eigendecomposition ...")
    dr=Taa1.shape[0]    
    da=Taa1.shape[1]    
    T = np.matrix(np.zeros([3,3],dtype=np.complex64)) 
    C = np.matrix(np.zeros([3,3],dtype=np.complex64)) 
    L1 = np.zeros([dr,da]) 
    L2 = np.zeros([dr,da])
    L3 = np.zeros([dr,da])
    U_11 = np.zeros([dr,da])
    U_21 = np.zeros([dr,da])
    U_31 = np.zeros([dr,da])
    U_12 = np.zeros([dr,da])
    U_22 = np.zeros([dr,da])
    U_32 = np.zeros([dr,da])
    U_13 = np.zeros([dr,da])
    U_23 = np.zeros([dr,da])
    U_33 = np.zeros([dr,da])
    T11 = np.zeros([dr,da])
    T22 = np.zeros([dr,da])
    T33 = np.zeros([dr,da])
    U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
    U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
    Uinv=U.getH() 
    for m in range(dr-1):
        print (np.round((m*100/dr),1),"%")
        for n in range(da-1):
            C[0,0] = Taa1[m,n]
            C[0,1] = Tab1[m,n] 
            C[0,2] = Tac1[m,n]
            C[1,0] = np.conj(Tab1[m,n])
            C[1,1] = Tbb1[m,n]
            C[1,2] = Tbc1[m,n]
            C[2,0] = np.conj(Tac1[m,n])
            C[2,1] = np.conj(Tbc1[m,n])
            C[2,2] = Tcc1[m,n]
            T=U.dot(C).dot(Uinv)   
            #
            T11[m][n]=T[0,0]
            T22[m][n]=T[1,1]
            T33[m][n]=T[2,2]
            #T = np.nan_to_num(T)
            eigenvalues, eigenvectors = np.linalg.eigh(T)
            
            ind = np.argsort(eigenvalues)# organize eigenvalues from higher to lower value
            eigenvalues[eigenvalues < 0] = 0    
            
            L1[m][n] = eigenvalues[ind[2]]
            L2[m][n] = eigenvalues[ind[1]]
            L3[m][n] = eigenvalues[ind[0]]
            
            U_11[m][n]=np.abs(eigenvectors[0,2]) 
            U_21[m][n]=np.abs(eigenvectors[1,2])
            U_31[m][n]=np.abs(eigenvectors[2,2])
     
            U_12[m][n]=np.abs(eigenvectors[0,1])
            U_22[m][n]=np.abs(eigenvectors[1,1])
            U_32[m][n]=np.abs(eigenvectors[2,1])
            
            U_13[m][n]=np.abs(eigenvectors[0,0])
            U_23[m][n]=np.abs(eigenvectors[1,0])
            U_33[m][n]=np.abs(eigenvectors[2,0])
    visRGB_from_T(T22, T33, T11, "RGB "+title1)
    return(L1,L2,L3,U_11,U_21,U_31,U_12,U_22,U_32,U_13,U_23,U_33)
    
def select_images_home(beam):
    if beam=="FQ19W":
        dateA="2014-06-05.rds2"
        dateB="2014-06-29.rds2"
        dateC="2014-07-23.rds2"
        dateD="2014-08-16.rds2"
        dateE="2014-09-09.rds2"
        paths=["D:\\Juanma\\"+beam+"\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+beam+"\\"+dateB+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateC+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateD+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateE+"\\"]
        dates=[dateA,dateB,dateC,dateD,dateE]
    elif beam=="FQ13W":
        dateA="2014-06-22.rds2"
        dateB="2014-07-16.rds2"
        dateC="2014-08-09.rds2"
        dateD="2014-09-02.rds2"
        dateE="2014-09-26.rds2" 
        paths=["D:\\Juanma\\"+beam+"\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+beam+"\\"+dateB+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateC+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateD+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateE+"\\"]  
        dates=[dateA,dateB,dateC,dateD,dateE]
    elif beam=="FQ8W":
        dateA="2014-05-22.rds2"
        dateB="2014-06-15.rds2"
        dateC="2014-07-09.rds2"
        dateD="2014-08-02.rds2"
        dateE="2014-08-26.rds2" 
        dateF="2014-09-19.rds2" 
        
        paths=["D:\\Juanma\\"+beam+"\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+beam+"\\"+dateB+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateC+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateD+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateE+"\\",
               "D:\\Juanma\\"+beam+"\\"+dateF+"\\"]  
        dates=[dateA,dateB,dateC,dateD,dateE,dateF]
    elif beam=="All": # beams 13 and 19
        dateA="2014-06-05.rds2"
        dateB="2014-06-22.rds2"
        dateC="2014-06-29.rds2"
        dateD="2014-07-16.rds2"
        dateE="2014-07-23.rds2"
        dateF="2014-08-09.rds2"
        dateG="2014-08-16.rds2"
        dateH="2014-09-02.rds2"
        dateI="2014-09-09.rds2"
        dateJ="2014-09-26.rds2"
        paths=["D:\\Juanma\\FQ19W\\"+dateA+"\\",   # home
               "D:\\Juanma\\FQ13W\\"+dateB+"\\",
               "D:\\Juanma\\FQ19W\\"+dateC+"\\",
               "D:\\Juanma\\FQ13W\\"+dateD+"\\",
               "D:\\Juanma\\FQ19W\\"+dateE+"\\",
               "D:\\Juanma\\FQ13W\\"+dateF+"\\",   
               "D:\\Juanma\\FQ19W\\"+dateG+"\\",
               "D:\\Juanma\\FQ13W\\"+dateH+"\\",
               "D:\\Juanma\\FQ19W\\"+dateI+"\\",
               "D:\\Juanma\\FQ13W\\"+dateJ+"\\"]
        dates=[dateA,dateB,dateC,dateD,dateE,dateF,dateG,dateH,dateI,dateJ]
    elif beam=="AllAll": # beams 8,13,19
        dateA="2014-05-22.rds2"
        dateB="2014-06-05.rds2"
        dateC="2014-06-15.rds2"
        dateD="2014-06-22.rds2"
        dateE="2014-06-29.rds2"
        dateF="2014-07-09.rds2"
        dateG="2014-07-16.rds2"
        dateH="2014-07-23.rds2"
        dateI="2014-08-02.rds2"
        dateJ="2014-08-09.rds2"
        dateK="2014-08-16.rds2"
        dateL="2014-08-26.rds2"
        dateM="2014-09-02.rds2"
        dateN="2014-09-09.rds2"
        dateO="2014-09-19.rds2"
        dateP="2014-09-26.rds2"
        paths=["D:\\Juanma\\FQ8W\\"+dateA+"\\",
               "D:\\Juanma\\FQ19W\\"+dateB+"\\",
               "D:\\Juanma\\FQ8W\\"+dateC+"\\",# home
               "D:\\Juanma\\FQ13W\\"+dateD+"\\",
               "D:\\Juanma\\FQ19W\\"+dateE+"\\",
               "D:\\Juanma\\FQ8W\\"+dateF+"\\",
               "D:\\Juanma\\FQ13W\\"+dateG+"\\",
               "D:\\Juanma\\FQ19W\\"+dateH+"\\",   
               "D:\\Juanma\\FQ8W\\"+dateI+"\\",
               "D:\\Juanma\\FQ13W\\"+dateJ+"\\",
               "D:\\Juanma\\FQ19W\\"+dateK+"\\",
               "D:\\Juanma\\FQ8W\\"+dateL+"\\",
               "D:\\Juanma\\FQ13W\\"+dateM+"\\",
               "D:\\Juanma\\FQ19W\\"+dateN+"\\",
               "D:\\Juanma\\FQ8W\\"+dateO+"\\",
               "D:\\Juanma\\FQ13W\\"+dateP+"\\"]
        dates=[dateA,dateB,dateC,dateD,dateE,dateF,dateG,dateH,dateI,dateJ,dateK,dateL,dateM,dateN,dateO,dateP]
    return(paths,dates)

def select_images_office(beam):
    comp="Seville 2014\\"
    if beam=="FQ19W":
        dateA="2014-06-05.rds2"
        dateB="2014-06-29.rds2"
        dateC="2014-07-23.rds2"
        dateD="2014-08-16.rds2"
        dateE="2014-09-09.rds2"
        paths=["D:\\Juanma\\"+comp+beam+"\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+comp+beam+"\\"+dateB+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateC+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateD+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateE+"\\"]
        dates=[dateA,dateB,dateC,dateD,dateE]
    elif beam=="FQ13W":
        dateA="2014-06-22.rds2"
        dateB="2014-07-16.rds2"
        dateC="2014-08-09.rds2"
        dateD="2014-09-02.rds2"
        dateE="2014-09-26.rds2" 
        paths=["D:\\Juanma\\"+comp+beam+"\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+comp+beam+"\\"+dateB+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateC+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateD+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateE+"\\"]  
        dates=[dateA,dateB,dateC,dateD,dateE]
    elif beam=="FQ8W":
        dateA="2014-05-22.rds2"
        dateB="2014-06-15.rds2"
        dateC="2014-07-09.rds2"
        dateD="2014-08-02.rds2"
        dateE="2014-08-26.rds2" 
        dateF="2014-09-19.rds2" 
        
        paths=["D:\\Juanma\\"+comp+beam+"\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+comp+beam+"\\"+dateB+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateC+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateD+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateE+"\\",
               "D:\\Juanma\\"+comp+beam+"\\"+dateF+"\\"]  
        dates=[dateA,dateB,dateC,dateD,dateE,dateF]
    elif beam=="All": # beams 13 and 19
        dateA="2014-06-05.rds2"
        dateB="2014-06-22.rds2"
        dateC="2014-06-29.rds2"
        dateD="2014-07-16.rds2"
        dateE="2014-07-23.rds2"
        dateF="2014-08-09.rds2"
        dateG="2014-08-16.rds2"
        dateH="2014-09-02.rds2"
        dateI="2014-09-09.rds2"
        dateJ="2014-09-26.rds2"
        paths=["D:\\Juanma\\"+comp+"\\FQ19W\\"+dateA+"\\",   # home
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateB+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateC+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateD+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateE+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateF+"\\",   
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateG+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateH+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateI+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateJ+"\\"]
        dates=[dateA,dateB,dateC,dateD,dateE,dateF,dateG,dateH,dateI,dateJ]
    elif beam=="AllAll": # beams 8,13,19
        dateA="2014-05-22.rds2"
        dateB="2014-06-05.rds2"
        dateC="2014-06-15.rds2"
        dateD="2014-06-22.rds2"
        dateE="2014-06-29.rds2"
        dateF="2014-07-09.rds2"
        dateG="2014-07-16.rds2"
        dateH="2014-07-23.rds2"
        dateI="2014-08-02.rds2"
        dateJ="2014-08-09.rds2"
        dateK="2014-08-16.rds2"
        dateL="2014-08-26.rds2"
        dateM="2014-09-02.rds2"
        dateN="2014-09-09.rds2"
        dateO="2014-09-19.rds2"
        dateP="2014-09-26.rds2"
        paths=["D:\\Juanma\\"+comp+"\\FQ8W\\"+dateA+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateB+"\\",
               "D:\\Juanma\\"+comp+"\\FQ8W\\"+dateC+"\\",# home
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateD+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateE+"\\",
               "D:\\Juanma\\"+comp+"\\FQ8W\\"+dateF+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateG+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateH+"\\",   
               "D:\\Juanma\\"+comp+"\\FQ8W\\"+dateI+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateJ+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateK+"\\",
               "D:\\Juanma\\"+comp+"\\FQ8W\\"+dateL+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateM+"\\",
               "D:\\Juanma\\"+comp+"\\FQ19W\\"+dateN+"\\",
               "D:\\Juanma\\"+comp+"\\FQ8W\\"+dateO+"\\",
               "D:\\Juanma\\"+comp+"\\FQ13W\\"+dateP+"\\"]
        dates=[dateA,dateB,dateC,dateD,dateE,dateF,dateG,dateH,dateI,dateJ,dateK,dateL,dateM,dateN,dateO,dateP]
    return(paths,dates)

def Multilook(img, win, outfile = [], flag = 0):        
    """
    This module filter the initial image with a defined kernel (e.g. a boxcar)
    and it returns only the central pixels (it does multilook).
    if Flag is 1 then it saves the result as a python file
    """
    kernel  = np.ones((win,win),np.float32)/(np.power(win,2))

    img_test =  signal.convolve2d(np.power(np.abs(img),2), kernel, 
                                  mode='full', boundary='fill', fillvalue=0)
    img_small = img_test[::win,::win]
    
    if flag == 1:
        np.save(outfile, img_small)
        img = 0
    
    return(img,img_small)
    

    
def training_change_matrices_all(change_matrices,coord_x,coord_y,dates):
        
    y_minc=int(min(coord_y[:,0]))
    y_maxc=int(max(coord_y[:,0]))
    x_minc=int(min(coord_x[:,0]))
    x_maxc=int(max(coord_x[:,0]))
    centroid1=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid1=centroid1.reshape([3*dates*dates,(centroid1.shape[3]*centroid1.shape[4])])
    centroid1=np.transpose(centroid1, (1,0))
    y_minc=int(min(coord_y[:,1]))
    y_maxc=int(max(coord_y[:,1]))
    x_minc=int(min(coord_x[:,1]))
    x_maxc=int(max(coord_x[:,1]))
    centroid2=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid2=centroid2.reshape([3*dates*dates,(centroid2.shape[3]*centroid2.shape[4])])
    centroid2=np.transpose(centroid2, (1,0))
    y_minc=int(min(coord_y[:,2]))
    y_maxc=int(max(coord_y[:,2]))
    x_minc=int(min(coord_x[:,2]))
    x_maxc=int(max(coord_x[:,2]))
    centroid3=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid3=centroid3.reshape([3*dates*dates,(centroid3.shape[3]*centroid3.shape[4])])
    centroid3=np.transpose(centroid3, (1,0))
    y_minc=int(min(coord_y[:,3]))
    y_maxc=int(max(coord_y[:,3]))
    x_minc=int(min(coord_x[:,3]))
    x_maxc=int(max(coord_x[:,3]))
    centroid4=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid4=centroid4.reshape([3*dates*dates,(centroid4.shape[3]*centroid4.shape[4])])
    centroid4=np.transpose(centroid4, (1,0))
    y_minc=int(min(coord_y[:,4]))
    y_maxc=int(max(coord_y[:,4]))
    x_minc=int(min(coord_x[:,4]))
    x_maxc=int(max(coord_x[:,4]))
    centroid5=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid5=centroid5.reshape([3*dates*dates,(centroid5.shape[3]*centroid5.shape[4])])
    centroid5=np.transpose(centroid5, (1,0))
    y_minc=int(min(coord_y[:,5]))
    y_maxc=int(max(coord_y[:,5]))
    x_minc=int(min(coord_x[:,5]))
    x_maxc=int(max(coord_x[:,5]))
    centroid6=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid6=centroid6.reshape([3*dates*dates,(centroid6.shape[3]*centroid6.shape[4])])
    centroid6=np.transpose(centroid6, (1,0))
    y_minc=int(min(coord_y[:,6]))
    y_maxc=int(max(coord_y[:,6]))
    x_minc=int(min(coord_x[:,6]))
    x_maxc=int(max(coord_x[:,6]))
    centroid7=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid7=centroid7.reshape([3*dates*dates,(centroid7.shape[3]*centroid7.shape[4])])
    centroid7=np.transpose(centroid7, (1,0))
    y_minc=int(min(coord_y[:,7]))
    y_maxc=int(max(coord_y[:,7]))
    x_minc=int(min(coord_x[:,7]))
    x_maxc=int(max(coord_x[:,7]))
    centroid8=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
    centroid8=centroid8.reshape([3*dates*dates,(centroid8.shape[3]*centroid8.shape[4])])
    centroid8=np.transpose(centroid8, (1,0))
    
    X=np.concatenate((centroid1,centroid2,centroid3,centroid4,centroid5,centroid6,centroid7,centroid8),axis=0)
    
    Y1=np.zeros([centroid1.shape[0]])
    Y2=np.zeros([centroid2.shape[0]])
    Y3=np.zeros([centroid3.shape[0]])
    Y4=np.zeros([centroid4.shape[0]])
    Y5=np.zeros([centroid5.shape[0]])
    Y6=np.zeros([centroid6.shape[0]])
    Y7=np.zeros([centroid7.shape[0]])
    Y8=np.zeros([centroid8.shape[0]])
    
    Y1[:]=0
    Y2[:]=1
    Y3[:]=2
    Y4[:]=3
    Y5[:]=4
    Y6[:]=5
    Y7[:]=6
    Y8[:]=7

    y=np.concatenate((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8),axis=0)
    return(X,y) 
    
def assign_color(classifier_res,dr,da,img_title):
    m=0
    n=0
    classifier_res_img=np.zeros([dr,da,3], dtype=int)
    print("Assigning colour to each of the classes ...")
    for m in range(dr):
        for n in range(da):
            #classifier_res[m,n]=np.argmin(y_kmeans[m,n])
            if classifier_res[m,n]==0.0:
                classifier_res_img[m,n] =(255, 0, 0) # Rice - Red
            elif classifier_res[m,n]==1.0:
                classifier_res_img[m,n] =(0, 0, 255)  # River - Blue
            elif classifier_res[m,n]==2.0:
                classifier_res_img[m,n] = (0,250,0) # Crop 1 - Green
            elif classifier_res[m,n]==3.0:
                classifier_res_img[m,n] = (196,145,2) # Crop 2 - 
            elif classifier_res[m,n]==4.0:
                classifier_res_img[m,n] = (75,83,32) # Crop 3
            elif classifier_res[m,n]==5.0:
                classifier_res_img[m,n] = (255,255,0) # Crop 4 - Yellow
            elif classifier_res[m,n]==6.0:
                classifier_res_img[m,n] = (153,153,153) # Road - Grey
            elif classifier_res[m,n]==7.0:
                classifier_res_img[m,n] = (255,255,255) # City - White
            elif classifier_res[m,n]==8.0: 
                classifier_res_img[m,n] = (0,0,0) # class paneles - brownish
            elif classifier_res[m,n]==9.0:
                classifier_res_img[m,n] = (51,51,51) # road rice - cyan
            elif classifier_res[m,n]==10.0:
                classifier_res_img[m,n] = (0,0,0) # class none - Black
            elif classifier_res[m,n]==11.0: 
                classifier_res_img[m,n] = (200,230,150) # 
            elif classifier_res[m,n]==12.0:
                classifier_res_img[m,n] = (250,100,50) # 
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(img_title)
    ax1.imshow(classifier_res_img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return# Turn down for faster convergence

def training_change_matrices(change_matrices,coord_x,coord_y,Number_images):
    #i=0
    #b=0
    k=coord_x.shape[1]
    change_matrices_avg=np.zeros([3,Number_images,Number_images,k],dtype=np.complex64)
    #for i in range(len(paths)):
#    y_minc=np.zeros([k],dtype=np.int)
#    y_maxc=np.zeros([k],dtype=np.int)
#    x_minc=np.zeros([k],dtype=np.int)
#    x_maxc=np.zeros([k],dtype=np.int)

    for b in range (k):
        y_minc=int(min(coord_y[:,b]))
        y_maxc=int(max(coord_y[:,b]))
        x_minc=int(min(coord_x[:,b]))
        x_maxc=int(max(coord_x[:,b]))
#        da=x_max-x_min
#        dr=y_max-y_min
        
        change_matrices1=change_matrices[:,:,:,y_minc:y_maxc,x_minc:x_maxc]
        for r in range(Number_images):
            for s in range(Number_images):
                change_matrices_avg[0,s,r,b]=np.mean(change_matrices1[0,s,r,:,:])
                change_matrices_avg[1,s,r,b]=np.mean(change_matrices1[1,s,r,:,:])
                change_matrices_avg[2,s,r,b]=np.mean(change_matrices1[2,s,r,:,:])
    #    a[0,:,:]=change_matrices[0,:,:,int(x1),int(y1)]
    #    a[1,:,:]=change_matrices[1,:,:,int(x1),int(y1)]
    #    a[2,:,:]=change_matrices[2,:,:,int(x1),int(y1)]
    #    
    centroid1=change_matrices_avg[:,:,:,0]
    centroid2=change_matrices_avg[:,:,:,1]
    centroid3=change_matrices_avg[:,:,:,2]
    centroid4=change_matrices_avg[:,:,:,3]
    centroid5=change_matrices_avg[:,:,:,4]
    centroid6=change_matrices_avg[:,:,:,5]
    centroid7=change_matrices_avg[:,:,:,6]
    centroid8=change_matrices_avg[:,:,:,7]  
    #centroid9=change_matrices_avg[:,:,:,8]   
#    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
#    ax10.set_title(title)
#    ax10.imshow(change_matrices_avg)
#    ax10.axis('off')
#    plt.tight_layout()
#    change_matrices_avg1[0,:,:]=change_matrices_avg[:,:,0]
#    change_matrices_avg1[1,:,:]=change_matrices_avg[:,:,1]
#    change_matrices_avg1[2,:,:]=change_matrices_avg[:,:,2]
        
    return(centroid1,centroid2,centroid3,centroid4,centroid5,centroid6,centroid7,centroid8)
    

    
def mask_change_matrix_ROI(Change_matrix,x,print_flag,crop_class):
        # From the polygon select the corner to subset the change matrix
    y_coord=np.array([x[0,0,1],x[0,1,1],x[0,2,1],x[0,3,1]]) 
    y_min=int(min(y_coord))
    y_max=int(max(y_coord))
    x_coord=np.array([x[0,0,0],x[0,1,0],x[0,2,0],x[0,3,0]])
    x_min=int(min(x_coord))
    x_max=int(max(x_coord))
        # subset the change matrix
    small_Change_matrix = Change_matrix[:,:,:,y_min:y_max,x_min:x_max]   
        # Update the coordinates of the polygon so that it corresponds to the small change matrix
    x[:,:,1]=x[:,:,1]-y_min
    x[:,:,0]=x[:,:,0]-x_min
    x_coord1=np.array([x[0,0,0],x[0,1,0],x[0,2,0],x[0,3,0]])
    y_coord1=np.array([x[0,0,1],x[0,1,1],x[0,2,1],x[0,3,1]])
    
    y_min=int(min(y_coord1))
    y_max=int(max(y_coord1))
    x_min=int(min(x_coord1))
    x_max=int(max(x_coord1))
        # Create a 2D mask with ones in the ROI and zeros outside
    mask = np.zeros_like(small_Change_matrix[0,0,0,:,:],dtype=np.uint8)   
    mask=cv2.fillPoly(mask, x, 1)
    mask1=np.full_like(small_Change_matrix, 0)
        # Create the same for the multidimensional array of the small change matrix
    for i in range(small_Change_matrix.shape[0]):
        for j in range(small_Change_matrix.shape[1]):
            for z in range(small_Change_matrix.shape[2]):
                mask1[i,j,z,:,:]=mask
    # apply the mask    
    masked_iRGB_Change_matrix = small_Change_matrix*mask1
    centroid_rice=masked_iRGB_Change_matrix
        # Turn each pixel into a feature as accepted by SKlearn
    centroid_rice1=centroid_rice.reshape([3*small_Change_matrix.shape[1]*small_Change_matrix.shape[2],(centroid_rice.shape[3]*centroid_rice.shape[4])])
    centroid_rice1=np.transpose(centroid_rice1, (1,0))
        # Remove the rows corresponding to the change matrices of the pixels outside the ROI (all elements in the row are zeros)
    centroid_rice2=centroid_rice1[~(centroid_rice1==0).all(1)]
        # Find the mean of the change matrices in the ROI
    centroid_rice3=np.mean(centroid_rice2,axis=0)
    centroid_rice4=centroid_rice3.reshape([3,small_Change_matrix.shape[1],small_Change_matrix.shape[2]])
    
    # Plot average change matrix of the ROI
    if print_flag == 1:
        fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
        ax10.set_title(crop_class)
        ax10.imshow(np.transpose(centroid_rice4, (1,2,0))*5)
        ax10.axis('off')
        plt.tight_layout()     
        plt.show()

    return(centroid_rice2)
    
def assign_color_agrisar(classifier_res,dr,da,img_title,mask):
    m=0
    n=0
    classifier_res_img=np.zeros([dr,da,3], dtype=int)
    print("Assigning colour to each of the classes ...")
    for m in range(dr):
        for n in range(da):
            #classifier_res[m,n]=np.argmin(y_kmeans[m,n])
            if classifier_res[m,n]==0.0:
                classifier_res_img[m,n] =(76, 153, 0) # Barley
            elif classifier_res[m,n]==1.0:
                classifier_res_img[m,n] =(255, 204, 204)  # Canary seed
            elif classifier_res[m,n]==2.0:
                classifier_res_img[m,n] = (0,204,204) # Canola
            elif classifier_res[m,n]==3.0:
                classifier_res_img[m,n] = (0,102,102) # Chemical fallow
            elif classifier_res[m,n]==4.0:
                classifier_res_img[m,n] = (0,255,0) # Durum wheat
            elif classifier_res[m,n]==5.0:
                classifier_res_img[m,n] = (255,255,0) # Field pea
            elif classifier_res[m,n]==6.0:
                classifier_res_img[m,n] = (255,0,0) # Flax
            elif classifier_res[m,n]==7.0:
                classifier_res_img[m,n] = (0,0,255) # Grass
            elif classifier_res[m,n]==8.0: 
                classifier_res_img[m,n] = (255,128,0) # Lentil
            elif classifier_res[m,n]==9.0:
                classifier_res_img[m,n] = (102,255,255) # Mixed Hay
            elif classifier_res[m,n]==10.0:
                classifier_res_img[m,n] = (255,51,255) # Mixed pasture
            elif classifier_res[m,n]==11.0: 
                classifier_res_img[m,n] = (153,153,0) # Oat
            elif classifier_res[m,n]==12.0:
                classifier_res_img[m,n] = (102,0,204) # Spring wheat
            elif classifier_res[m,n]==13.0:
                classifier_res_img[m,n] = (160,160,160) # Summer fallow
            elif classifier_res[m,n]==14.0:
                classifier_res_img[m,n] = (0,0,0) # no crop                

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==0:
                classifier_res_img[i,j]=(0,0,0)
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(img_title)
    ax1.imshow(classifier_res_img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return# Turn down for faster convergence    

def bayes_search_hyperparameters(model,No_iterations,X_train, y_train,X_test, y_test,save,path,model_name):
    """
    Created on Friday March 22 17:13 2019
    find the best combination of hyperparameters using the Bayes search method implemented with SK optimse
    So far includes KNN, random forest and NN
    Saves and returns the the optimal model 
    @author: Cristian Silva
    """
    t0 = time.time()
    if model=="KNN":
        print("Training KNN ...")
        estimator = KNeighborsClassifier(n_jobs=-1)
        parameters={'n_neighbors':(1,15)}
    elif model == "RF":
        print("Training Random forest ...")
        parameters={'n_estimators':(50,200), 'criterion':('gini','entropy')}
        estimator = RandomForestClassifier(n_jobs=-1)
    elif model == "MLP":    
        print("Training Neural Network...")
        parameters={'hidden_layer_sizes':[(50,50,50), (50,100,50), (100,),(50,100,100,50,25)],
                    'activation':('tanh', 'relu'),'solver':('sgd', 'adam'),'alpha': (0.0001, 0.05)}
        estimator = MLPClassifier(max_iter=600)
    
    opt = BayesSearchCV(estimator,search_spaces=parameters,optimizer_kwargs=None, n_iter=No_iterations, 
                    scoring=None, fit_params=None, n_jobs=-1, n_points=1, iid=True, refit=True, cv=5, 
                    verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score='raise', 
                    return_train_score=False)
    # callback handler
    def on_step(optim_result):
        score = opt.best_score_
        print("best score: %s" % score)
        #if score >= 0.98:
            #print('Interrupting!')
            #return True
    opt.fit(X_train, y_train, callback=on_step)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    
    print("Best estimator: "+str(opt.best_estimator_)) 
    print("Best parameters: "+str(opt.best_params_) )
    y_pred = opt.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"% (opt, metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
    run_time = (time.time() - t0)/60
    print('Training done in %.3f minutes' % run_time)
    if save == 1:
        filename = path+model_name+'.sav'
        pickle.dump(opt, open(filename, 'wb'))
    
    return(opt) # 
    
def read_parcel_coord(path,sheet,parcel):
    xls = pd.ExcelFile(path)
    coord_df = pd.read_excel(xls, sheet)
    parcel_coordinates=coord_df[parcel].values
    
    x1=int(parcel_coordinates[0])
    y1=int(parcel_coordinates[1])
    x2=int(parcel_coordinates[2])
    y2=int(parcel_coordinates[3])
    x3=int(parcel_coordinates[4])
    y3=int(parcel_coordinates[5])
    x4=int(parcel_coordinates[6])
    y4=int(parcel_coordinates[7])
    return(x1,y1,x2,y2,x3,y3,x4,y4)
    
def get_Single_Parcel(path1):
    """
    Shows an RGB image in Pauli basis to select a ROI
    Then shows the ROI to select the specific coordinates of the parcel
""" 
    print("Opening RGB composite for you to select the ROI with 4 clicks ...")    
    print("Opening C11 ...")    
    C11 = Open_C_diag_element(path1 + "\\C11.bin")
    print("Opening C22 ...")    
    C22 = Open_C_diag_element(path1 + "\\C22.bin")
    print("Opening C33 ...")    
    C33 = Open_C_diag_element(path1 + "\\C33.bin")
   
    iRGB= visRGB(C11,C22,C33,'RGB - Full Image')
        #Get ROI coordinates and plot it
    print('please click four points')
    Coordinates = plt.ginput(4)
    x1,y1=Coordinates[0]
    x2,y2=Coordinates[1]
    x3,y3=Coordinates[2]
    x4,y4=Coordinates[3]
    
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    x3=int(x3)
    x4=int(x4)
    y3=int(y3)
    y4=int(y4)

    y=np.array([y1,y2,y3,y4])
    x=np.array([x1,x2,x3,x4])
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    
    plt.close()           
    iRGB_ROI = iRGB[y_min:y_max,x_min:x_max]
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
    ax1.set_title('Region of interest')
    ax1.imshow(iRGB_ROI)  
    
    #Get parcel coordinates and plot it
    print('please click four points')
    Coordinates = plt.ginput(4)
    x1,y1=Coordinates[0]
    x2,y2=Coordinates[1]
    x3,y3=Coordinates[2]
    x4,y4=Coordinates[3]
    
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    x3=int(x3)
    x4=int(x4)
    y3=int(y3)
    y4=int(y4)
    y=np.array([y1+y_min,y2+y_min,y3+y_min,y4+y_min])
    x=np.array([x1+x_min,x2+x_min,x3+x_min,x4+x_min])
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    y1=y[0]
    y2=y[1]
    y3=y[2]
    y4=y[3]
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    return(x1,y1,x2,y2,x3,y3,x4,y4,x,y)

def get_ROI():
    print('please click four points')
    Coordinates = plt.ginput(4)
    x1,y1=Coordinates[0]
    x2,y2=Coordinates[1]
    x3,y3=Coordinates[2]
    x4,y4=Coordinates[3]
    y=np.array([y1,y2,y3,y4])
    x=np.array([x1,x2,x3,x4])    
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    x3=int(x3)
    x4=int(x4)
    y3=int(y3)
    y4=int(y4)
    return(x1,y1,x2,y2,x3,y3,x4,y4,x,y)
    
def mask_only_ROI(C11,C12,C13,C22,C23,C33,x,y):
    print("masking out to consider only the ROI...")
    mask = np.full_like(C11, 0) #create the mask = array of zeros with the original image size
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]]) #roi_corners = Array with the Parcel coordinates
    #channel_count = C11.shape[2]  # Number of channels (3)
    channel_count = 1  # ones
    ignore_mask_color = (1)*channel_count # How many channels has the image for filling them
#    cv2.fillPoly(mask, roi_corners, ignore_mask_color) # Fill of ones the polygon with the parcel coordinates (put a mask)
    # apply the mask    
    masked_iRGB_C11 = C11*mask# Merge the original image with the mask (Only leave the pixels inside the polygon)
    masked_iRGB_C12 = C12*mask# Merge the original image with the mask (Only leave the pixels inside the polygon)
    masked_iRGB_C13 = C13*mask# Merge the original image with the mask (Only leave the pixels inside the polygon)
    masked_iRGB_C22 = C22*mask# Merge the original image with the mask (Only leave the pixels inside the polygon)
    masked_iRGB_C23 = C23*mask# Merge the original image with the mask (Only leave the pixels inside the polygon)
    masked_iRGB_C33 = C33*mask# Merge the original image with the mask (Only leave the pixels inside the polygon)

    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    masked_iRGB_C11 = masked_iRGB_C11[y_min:y_max,x_min:x_max]    
    masked_iRGB_C12 = masked_iRGB_C12[y_min:y_max,x_min:x_max]    
    masked_iRGB_C13 = masked_iRGB_C13[y_min:y_max,x_min:x_max]    
    masked_iRGB_C22 = masked_iRGB_C22[y_min:y_max,x_min:x_max]    
    masked_iRGB_C23 = masked_iRGB_C23[y_min:y_max,x_min:x_max]    
    masked_iRGB_C33 = masked_iRGB_C33[y_min:y_max,x_min:x_max]    
    return(masked_iRGB_C11,masked_iRGB_C12,masked_iRGB_C13,masked_iRGB_C22,masked_iRGB_C23,masked_iRGB_C33)

class eigendecompositions():
    """
    Use this class to perform eigendecomposition of coherency matrices
    Depending on the application, the 
    Return: List with numpy arrays of ...
    """
    def __init__(self, application):
        self.application = application
        self.epsilon = 1e-7
    
    def gral_eigendecomposition(self, Tc):
        """
        Input: A coherency matrix. Could also be difference or ratio matrix
        """
        print("Eigendecomposition ...")
        #Tc = T22 - T11
        eigenvalues, eigenvectors = np.linalg.eigh(Tc)
        
        #ind = np.argsort(eigenvalues)# organize eigenvalues from higher to lower value
        #eigenvalues[eigenvalues < 0] = 0 
        
        #L1 = eigenvalues[ind[2]] # already come sorted from np.linalg.eigh
        #L2 = eigenvalues[ind[1]]
        #L3 = eigenvalues[ind[0]]
        
        L1 = eigenvalues[:,:,2]
        L2 = eigenvalues[:,:,1]
        L3 = eigenvalues[:,:,0]
        
        U_11=np.abs(eigenvectors[:,:,0,2]) 
        U_21=np.abs(eigenvectors[:,:,1,2])
        U_31=np.abs(eigenvectors[:,:,2,2])
         
        U_12=np.abs(eigenvectors[:,:,0,1])
        U_22=np.abs(eigenvectors[:,:,1,1])
        U_32=np.abs(eigenvectors[:,:,2,1])
        
        U_13=np.abs(eigenvectors[:,:,0,0])
        U_23=np.abs(eigenvectors[:,:,1,0])
        U_33=np.abs(eigenvectors[:,:,2,0])  
        
        if self.application =="Single image":
            L1_inc = np.where(L1>0,L1,0)
            L2_inc = np.where(L2>0,L2,0)
            L3_inc = np.where(L3>0,L3,0)
            
            B = (L1_inc*U_11) + (L2_inc*U_12) + (L3_inc*U_13) # https://earth.esa.int/documents/653194/656796/Polarimetric_Decompositions.pdf
            R = (L1_inc*U_21) + (L2_inc*U_22) + (L3_inc*U_23)
            G = (L1_inc*U_31) + (L2_inc*U_32) + (L3_inc*U_33)
            list_=[R,G,B]
            
        if self.application =="Difference change detection":
            L1_inc = np.where(L1>0,L1,0)
            L2_inc = np.where(L2>0,L2,0)
            L3_inc = np.where(L3>0,L3,0)
            self.L1_inc=L1_inc
            self.L2_inc=L2_inc
            self.L3_inc=L3_inc
            B_inc = (L1_inc*U_11) + (L2_inc*U_12) + (L3_inc*U_13)
            R_inc = (L1_inc*U_21) + (L2_inc*U_22) + (L3_inc*U_23)
            G_inc = (L1_inc*U_31) + (L2_inc*U_32) + (L3_inc*U_33)
            
            L1_dec = np.where(L1<0,L1,0)
            L2_dec = np.where(L2<0,L2,0)
            L3_dec = np.where(L3<0,L3,0)
            self.L1_dec=L1_dec
            self.L2_dec=L2_dec
            self.L3_dec=L3_dec  
            B_dec = (L1_dec*U_11) + (L2_dec*U_12) + (L3_dec*U_13)
            R_dec = (L1_dec*U_21) + (L2_dec*U_22) + (L3_dec*U_23)
            G_dec = (L2_dec*U_31) + (L2_dec*U_32) + (L3_dec*U_33)

            list_=[R_inc, G_inc, B_inc, R_dec, G_dec, B_dec]
            
        if self.application == "Ratio change detection":
            L1_inc = np.where(L1>1,L1,0)
            L2_inc = np.where(L2>1,L2,0)
            L3_inc = np.where(L3>1,L3,0)
            self.L1_inc=L1_inc
            self.L2_inc=L2_inc
            self.L3_inc=L3_inc
            B_inc = (L1_inc*U_11) + (L2_inc*U_12) + (L3_inc*U_13)
            R_inc = (L1_inc*U_21) + (L2_inc*U_22) + (L3_inc*U_23)
            G_inc = (L1_inc*U_31) + (L2_inc*U_32) + (L3_inc*U_33)
            
            L1_dec = np.where(L1<1,L1,0)
            L2_dec = np.where(L2<1,L2,0)
            L3_dec = np.where(L3<1,L3,0)
            self.L1_dec=L1_dec
            self.L2_dec=L2_dec
            self.L3_dec=L3_dec  
            B_dec = (L1_dec*U_11) + (L2_dec*U_12) + (L3_dec*U_13)
            R_dec = (L1_dec*U_21) + (L2_dec*U_22) + (L3_dec*U_23)
            G_dec = (L2_dec*U_31) + (L2_dec*U_32) + (L3_dec*U_33)
            
            #        R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:] = R_inc, G_inc, B_inc
            #        R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:] = R_dec, G_dec, B_dec
            list_=[R_inc, G_inc, B_inc, R_dec, G_dec, B_dec]
        
        self.L1=L1
        self.L2=L2
        self.L3=L3
        self.U_11=U_11
        self.U_12=U_12
        self.U_13=U_13
        self.U_21=U_21
        self.U_22=U_22
        self.U_23=U_23   
        self.U_31=U_31
        self.U_32=U_32
        self.U_33=U_33
        return list_
    
    def vis(self,L1_change,L2_change,L3_change,add_or_remove):
        den = (np.abs(self.L1)+np.abs(self.L2)+np.abs(self.L3)+self.epsilon)
        if add_or_remove == 'added':
            P1=L1_change/den
            #P1 = np.where(L1_change < 0,0,P1)
            P2=L2_change/den
            #P2 = np.where(L2_change < 0,0,P2)
            P3=L3_change/den
            #P3 = np.where(L3_change < 0,0,P3)
                
        if add_or_remove == 'removed':
            P1=np.abs(L1_change)/den
            #P1 = np.where(L1_change > 0,0,P1)
            P2=np.abs(L2_change)/den
            #P2 = np.where(L2_change > 0,0,P2)
            P3=np.abs(L3_change)/den
            #P3 = np.where(L3_change > 0,0,P3)
                
        alpha1=np.arccos(self.U_11)
        alpha2=np.arccos(self.U_12)
        alpha3=np.arccos(self.U_13)
        
        beta1=np.arccos((self.U_21)/np.sin(alpha1))
        beta2=np.arccos((self.U_22)/np.sin(alpha2))
        beta3=np.arccos((self.U_23)/np.sin(alpha3))
        
        L_avg=(P1*np.abs(L1_change))+(P2*np.abs(L2_change))+(P3*np.abs(L3_change))
        alpha_avg=(P1*alpha1)+(P2*alpha2)+(P3*alpha3)
        beta_avg =(P1*beta1) +(P2*beta2) +(P3*beta3)
        
        R=(np.sqrt(np.abs(L_avg)))*(np.sin(alpha_avg))*(np.cos(beta_avg)) 
        G=(np.sqrt(np.abs(L_avg)))*(np.sin(alpha_avg))*(np.sin(beta_avg)) 
        B=(np.sqrt(np.abs(L_avg)))*(np.cos(alpha_avg))  
#        title=""
#        factor=3              
#        visRGB_from_T(R, G, B,title,factor=factor)
        return (R,G,B)
   
class change_matrix_of_stack:
    def __init__(self, N, folder1, dates2, ROI_size, header,
                 datatype = 'float32', basis="P",in_format="img",application = "Difference change detection"):
        self.datatype = datatype
        self.basis=basis    # Lexicographic (L) or Pauli (P)
        self.in_format=in_format # Bin or img
        self.N=N
        self.folder1=folder1
        self.dates2 = dates2
        self.ROI_size=ROI_size
        self.header = header
        self.application = application
        self.eigen = eigendecompositions(self.application)
    def call(self):
        # Empty array to save outputs of multitemporal change detection
        x_max = self.ROI_size[3]
        y_max = self.ROI_size[1]
        R_avg=np.zeros([self.N,self.N,x_max,y_max])
        G_avg=np.zeros([self.N,self.N,x_max,y_max])
        B_avg=np.zeros([self.N,self.N,x_max,y_max])        
        for img in range(self.N): ####### open image of date i and fix it   
            print(str(img+1))
            """
            ####################################### T11 ###############################
            """
            a=[]
            for file in os.listdir(self.folder1):
                if file.endswith(".img"):
                    if self.dates2[img] in file:
                        a.append(file)    
            
            T11=array2D_of_coherency_matrices_from_stack_SNAP(self.folder1,self.basis,self.in_format,
                                                                  self.ROI_size,self.header,self.datatype,a)
            """
            ####################################### T22 ###############################
            """    
            for img_2 in range(self.N): ####### open image of date j, looping from zero to the nth image     
                a=[]
                for file in os.listdir(self.folder1):
                    if file.endswith(".img"):
                        if self.dates2[img_2] in file:
                            a.append(file) 
                T22=array2D_of_coherency_matrices_from_stack_SNAP(self.folder1,self.basis,self.in_format,
                                                                      self.ROI_size,self.header,self.datatype,a)
                """
                ######################### Bi-Date quad pol Change detection ################
                """
                print("")
                print("Row "+str(img+1)+", Column "+ str(img_2+1))
                print("Processing added scattering mechanisms...")
                Tc = (T22 - T11)      
                List_RGB = self.eigen.gral_eigendecomposition(Tc) # to store eigendecomposition results in the class
                R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:] = self.eigen.vis(self.eigen.L1_inc,self.eigen.L2_inc,self.eigen.L3_inc,add_or_remove='added') # added SMs
                R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:] = self.eigen.vis(self.eigen.L1_dec,self.eigen.L2_dec,self.eigen.L3_dec,add_or_remove='removed') # removed SMs
                
                # Alberto (no log though)
                #if application == "Difference change detection" or application == "Ratio change detection":
                #R_avg[img,img_2,:,:],G_avg[img,img_2,:,:],B_avg[img,img_2,:,:] = List_RGB[0],List_RGB[1],List_RGB[2] # see the gral_eigendecomposition to understand the colors
                #R_avg[img_2,img,:,:],G_avg[img_2,img,:,:],B_avg[img_2,img,:,:] = List_RGB[3],List_RGB[4],List_RGB[5]
        return(R_avg,G_avg,B_avg)