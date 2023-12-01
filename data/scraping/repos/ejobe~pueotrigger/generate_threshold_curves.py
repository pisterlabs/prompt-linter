import numpy
from scipy.optimize import curve_fit
import coherent_sum
import tools.aso_geometry as aso_geometry

import myplot
import matplotlib.pyplot as plt


def generatePowerSums(noise, window=32, step=16, save=True):
    power = coherent_sum.powerSum(noise, window, step)
    if save:
        numpy.save('power_'+str(window)+'_'+str(step)+'.npy', power[0])

    return power


if __name__=='__main__':
    import sys

    #uncomment to generate power sums from noise voltages
    #
    '''
    noise_data = numpy.load('/scratch/midway2/ejo/aso_noise/simulated_noise_2pow26fbins_v1.npy')
    noise_data = noise_data.flatten()
    power = generatePowerSums(numpy.real(noise_data), window=32, step=16)
    sys.exit()
    '''
    #
    ###

    #generate threshold curves:
    power_files=[]
    power_files.append('/scratch/midway2/ejo/aso_noise/power_32_16_v1.npy')
    power_files.append('/scratch/midway2/ejo/aso_noise/power_16_8_v1.npy')
    power_files.append('/scratch/midway2/ejo/aso_noise/power_32_16_v2.npy')
    power_files.append('/scratch/midway2/ejo/aso_noise/power_16_8_v2.npy')
    power_sampling_interval = [16,8,16,8]
    power_sampling_window = [32,16,32,16]
    power_fit_region=[[1.e1, 1.e4],[1.e1, 1.e4],[1.e1, 1.e4],[1.e1, 1.e4]] #choose region of rate vs. threshold curve to perform power-law fit
    
    thresh_array = numpy.arange(0.2,7.,0.15)

    plot_label = ['Profile 1, Window N=32', 'Profile 1, Window N=16','Profile 2, Window N=32', 'Profile 2, Window N=16']
    plot_colors = ['blue', 'lightblue', 'green', 'lightgreen']
    fit_colors = ['darkred', 'darkred', 'black', 'black']
    
    for i, f in enumerate(power_files):

        powersums = numpy.load(f) 
        
        hits=[]
        
        for thresh in thresh_array:
            hits.append(len(numpy.where(powersums >= thresh)[0]))

        hits = numpy.array(hits)
        
        #START fit:
        indx=numpy.where(hits > 1)[0]
        #print indx
        indx_fit=numpy.where((hits > power_fit_region[i][0] ) & (hits < power_fit_region[i][1]))[0] # fine tuned
        #print indx_fit
        ##fit using polyfit in semi-log space:
        logy = numpy.log(hits[indx_fit]/(power_sampling_interval[i] * aso_geometry.ritc_sample_step * 1e-9 * len(powersums)))
        logy_w = numpy.log(5./(numpy.sqrt(hits[indx_fit]/(power_sampling_interval[i] * aso_geometry.ritc_sample_step * 1e-9 * len(powersums)))))
        coeff=numpy.polyfit(thresh_array[indx_fit], logy, w=logy_w, deg=1)
        poly = numpy.poly1d(coeff)
        #yfit = lambda x : numpy.power(10, poly(thresh_array[indx_fit]))
        yfit = lambda x : numpy.exp(poly(thresh_array[indx]))
        yfit_extended = lambda x1 : numpy.exp(poly(thresh_array))
        #END fit
        plt.figure(1)
        plt.errorbar(thresh_array, hits, yerr=5./(numpy.sqrt(hits)), elinewidth=3, fmt='o', ms=3,c=plot_colors[i])
        plt.yscale('log')

        plt.figure(2, figsize=(8,8))
        plt.errorbar(thresh_array, hits/(power_sampling_interval[i] * aso_geometry.ritc_sample_step * 1e-9 * len(powersums)),
                                                      yerr=5./(numpy.sqrt(hits/(power_sampling_interval[i] * aso_geometry.ritc_sample_step * 1e-9 * len(powersums)))),
                     elinewidth=3, c=plot_colors[i], fmt='o', ms=5, label=plot_label[i])
        
        plt.plot(thresh_array, yfit_extended(thresh_array), '--', color=fit_colors[i], lw=1)

        plt.figure(3)
        plt.hist(powersums, bins=thresh_array, alpha=0.5)
        plt.yscale('log')

    plt.figure(2)
    plt.ylim([.01,1e9])
    plt.xlim([0,8])
    plt.xlabel('Power Threshold = P / (N $\sigma^{2}$)')
    plt.ylabel('Rate [Hz]')
    plt.grid(True)
    plt.legend(numpoints=1)
    plt.yscale('log')
    plt.show()
            

    
