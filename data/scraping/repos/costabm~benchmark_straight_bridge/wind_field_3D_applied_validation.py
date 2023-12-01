# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 12:08:57 2018

@author: bernardc

This script runs wind_field_3D.py function, tests, and hopefully validates it
To reduce the calculation time / memory, reduce: num_nodes, T, sample_freq.
"""

def wind_field_3D_applied_validation_func(g_node_coor, windspeed, dt, wind_block_T, beta_DB, arc_length, R,
                                          Ii_simplified_bool, f_min, f_max, n_freq, n_nodes_validated, node_test_S_a,
                                          n_nodes_val_coh, export_folder=r"wind_field\data\plots"):

    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    from buffeting import beta_0_func, theta_0, U_bar_func, g_elem_nodes_func, Ii_func, S_a_nondim_func, Cij_func, iLj_func
    from transformations import T_GsGw_func
    from pathlib import Path

    Path(rf"{export_folder}").mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

    # Time domain
    n_windpoints = len(windspeed[0,0])
    wind_T = (n_windpoints - 1) * dt
    wind_freq = 1/dt
    time_array = np.linspace(0, wind_T, n_windpoints)

    # Frequency domain
    f_array = np.linspace(f_min, f_max, n_freq)

    g_node_num = len(g_node_coor)
    g_nodes = np.array(list(range(g_node_num)))  # starting at 0
    g_elem_nodes = g_elem_nodes_func(g_nodes)
    g_elem_num = g_node_num - 1
    g_elem_L_3D = np.array([np.linalg.norm(g_node_coor[g_elem_nodes[i, 1]] - g_node_coor[g_elem_nodes[i, 0]]) for i in range(g_elem_num)])
    g_s_3D = np.array([0] + list(np.cumsum(g_elem_L_3D)))

    beta_0 = beta_0_func(beta_DB)
    T_GsGw = T_GsGw_func(beta_0, theta_0)
    U_bar = U_bar_func(g_node_coor)
    U_bar_avg = (U_bar[:, None] + U_bar) / 2  # [m * n] matrix
    Ii = Ii_func(g_node_coor, beta_DB, Ii_simplified_bool)
    Cij = Cij_func(cond_rand_C=False)
    iLj = iLj_func(g_node_coor)
    iLj_avg = (iLj[:, None, :, :] + iLj) / 2

    windspeed_u = windspeed[1]
    windspeed_v = windspeed[2]
    windspeed_w = windspeed[3]

    S_a_nondim = S_a_nondim_func(g_node_coor, f_array, plot_S_a_nondim=False)

    arc_angle = arc_length / R  # rad. "Aperture" angle of the whole bridge arc.
    chord = np.sin(arc_angle / 2) * R * 2
    sagitta = R - np.sqrt(R ** 2 - (chord / 2) ** 2)

    node_coor_wind = np.einsum('ni,ij->nj', g_node_coor, T_GsGw)[:n_nodes_validated]

    nodes_x = node_coor_wind[:, 0]  # in wind flow coordinates! (along flow)
    nodes_y = node_coor_wind[:, 1]  # in wind flow coordinates! (horizontal-across flow)
    nodes_z = node_coor_wind[:, 2]  # in wind flow coordinates! (usually-vertical flow)

    delta_x = np.abs(nodes_x[:, np.newaxis] - nodes_x)[:n_nodes_validated]  # [n * m] matrix. in wind flow coordinates! (along flow)
    delta_y = np.abs(nodes_y[:, np.newaxis] - nodes_y)[:n_nodes_validated]  # [n * m] matrix. in wind flow coordinates! (horizontal-across flow)
    delta_z = np.abs(nodes_z[:, np.newaxis] - nodes_z)[:n_nodes_validated]  # [n * m] matrix. in wind flow coordinates! (usually-vertical flow)

    U_bar_avg = U_bar_avg[:n_nodes_validated,:n_nodes_validated]
    U_bar = U_bar[:n_nodes_validated]

    # =============================================================================↨
    #  Plotting MOVIE of wind across the g_nodes.
    #  For this to work, run in the cmd: conda install -c conda-forge ffmpeg
    # =============================================================================
    # from matplotlib.animation import FuncAnimation
    # import matplotlib
    # matplotlib.use("Agg")
    # #  from matplotlib.animation import FFMpegWriter
    # fig, ax = plt.subplots()
    # xdata1, ydata1 = [], []
    # xdata2, ydata2 = [], []
    # xdata3, ydata3 = [], []
    # ln1, = plt.plot([], [], animated=True, label='U')
    # ln2, = plt.plot([], [], animated=True, label='v')
    # ln3, = plt.plot([], [], animated=True, label='w')
    # metadata = dict(title='Movie U', artist='Matplotlib', comment='Movie support!')
    # Writer = matplotlib.animation.writers['ffmpeg']
    # writer = Writer(fps=sample_freq, metadata=metadata)
    # def init():
    #    ax.set_xlim(0, n_nodes_val)
    #    ax.set_ylim(-30, 100)
    #    return ln1,
    #    return ln2,
    #    return ln3,
    # def update(frame):
    #    xdata1 = list(range(n_nodes_val))
    #    xdata2 = list(range(n_nodes_val))
    #    xdata3 = list(range(n_nodes_val))
    #    ydata1 = windspeed[0,:n_nodes_val,frame] # U
    #    ydata2 = windspeed_v[:n_nodes_val,frame] # v
    #    ydata3 = windspeed_w[:n_nodes_val,frame] # w
    #    ln1.set_data(xdata1, ydata1)
    #    ln2.set_data(xdata2, ydata2)
    #    ln3.set_data(xdata3, ydata3)
    #    return ln1,
    #    return ln2,
    #    return ln3,
    # plt.title('Generated wind speed across the bridge g_nodes')
    # plt.xlabel('Node number [-]')
    # plt.ylabel('Wind speed [m/s]')
    # plt.grid()
    # plt.legend()
    # ani = FuncAnimation(fig, update, frames=int(min(sample_T*sample_freq,300)), init_func=init, blit=True)
    # ani.save('wind_field\wind-per-node.mp4', writer=writer)
    # plt.close()

    # =============================================================================
    # Plotting wind direction
    # =============================================================================
    plt.figure()
    plt.title('wind direction')
    plt.plot(g_node_coor[:, 0], g_node_coor[:, 1])
    plt.arrow(chord / 2, -sagitta * 1.5, -500 * np.sin(beta_0), 500 * np.cos(beta_0), head_width=300, head_length=300,
              head_starts_at_zero=False)
    plt.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    plt.xlim([0, chord])
    plt.ylim([-chord / 2, chord / 2])
    plt.savefig(rf'{export_folder}\wind_direction')
    plt.close()

    # =============================================================================
    # Plotting wind speed in 3 g_nodes
    # =============================================================================
    plt.figure()
    plt.plot(time_array, windspeed[0, 0, :], label='s = '+str(np.round(g_s_3D[0]))+' m')
    plt.plot(time_array, windspeed[0, 1, :], label='s = '+str(np.round(g_s_3D[1]))+' m')
    plt.plot(time_array, windspeed[0, -1, :], label='s = '+str(np.round(g_s_3D[-1]))+' m')
    plt.legend(title='Position along arc:')
    plt.xlabel('time [s]')
    plt.ylabel('Wind speed [m/s]')
    plt.xlim([0, 120])
    plt.grid()
    plt.savefig(rf'{export_folder}\wind_speed_at_3_nodes')
    plt.close()

    # =============================================================================
    # Plotting the means of wind velocities
    # =============================================================================
    plt.figure()
    plt.plot(list(range(n_nodes_validated)), np.mean(windspeed[0, :n_nodes_validated, :], axis=1), 'o', color='red', label='mean(U)', alpha=0.4, linewidth=2)
    plt.plot(U_bar[:n_nodes_validated], 'o', color='green', label='target', alpha=0.5)
    plt.xlabel('Node number')
    plt.ylabel('Wind speed [m/s]')
    # plt.ylim([min(U_bar) * 0.99, max(U_bar) * 1.01])
    plt.legend()
    plt.grid()
    plt.savefig(rf'{export_folder}\wind_speed_means')
    plt.close()

    # =============================================================================
    # Plotting standard deviations of wind velocity
    # =============================================================================
    plt.figure()
    plt.scatter(list(range(n_nodes_validated)), np.std(windspeed_u[:n_nodes_validated, :], axis=1), color='blue', label='std(u)')
    plt.plot(Ii[:n_nodes_validated, 0] * U_bar[:n_nodes_validated], color='blue', label='target std(u)')
    plt.scatter(list(range(n_nodes_validated)), np.std(windspeed_v[:n_nodes_validated, :], axis=1), color='orange', label='std(v)')
    plt.plot(Ii[:n_nodes_validated, 1] * U_bar[:n_nodes_validated], color='orange', label='target std(v)')
    plt.scatter(list(range(n_nodes_validated)), np.std(windspeed_w[:n_nodes_validated, :], axis=1), color='green', label='std(w)')
    plt.plot(Ii[:n_nodes_validated, 2] * U_bar[:n_nodes_validated], color='green', label='target std(w)')
    plt.legend()
    plt.gca().set_ylim(bottom=0)
    plt.xlabel('Node number [-]')
    plt.ylabel('std.(wind speed) [m/s]')
    plt.grid()
    plt.savefig(rf'{export_folder}\wind_speed_standard-deviations')
    plt.close()

    # =============================================================================
    # Plotting auto-spectra - Non-dimensional
    # =============================================================================
    # wind_block_T = min(600, wind_T)  # s. Duration of each segment, to build an average in the Welch method
    # nperseg = len(windspeed_u[node_test_S_a]) / round(wind_T / wind_block_T)
    nperseg = int(len(windspeed_u[node_test_S_a]) / 20)
    u_1_freq, u_1_ps = signal.welch(windspeed_u[node_test_S_a], wind_freq, nperseg=nperseg)
    v_1_freq, v_1_ps = signal.welch(windspeed_v[node_test_S_a], wind_freq, nperseg=nperseg)
    w_1_freq, w_1_ps = signal.welch(windspeed_w[node_test_S_a], wind_freq, nperseg=nperseg)
    plt.figure(figsize=(5, 4), dpi=400)
    plt.title('$f\/\/S_i(f)/\sigma_i^2$')  # Dimensional f
    plt.plot(f_array, S_a_nondim[:, node_test_S_a, 0], color='blue', label='u', alpha=0.6)
    plt.plot(f_array, S_a_nondim[:, node_test_S_a, 1], color='orange', label='v', alpha=0.6)
    plt.plot(f_array, S_a_nondim[:, node_test_S_a, 2], color='brown', label='w', alpha=0.6)
    # the last scatter point is always smaller (somehow) so it is discared
    plt.plot(u_1_freq[:-1], u_1_ps[:-1] / ((Ii[:, 0][node_test_S_a] * U_bar[node_test_S_a]) ** 2) * u_1_freq[:-1], '--', color='blue', alpha=0.3, linewidth=0.4)
    plt.plot(v_1_freq[:-1], v_1_ps[:-1] / ((Ii[:, 1][node_test_S_a] * U_bar[node_test_S_a]) ** 2) * v_1_freq[:-1], '--', color='orange',alpha=0.3, linewidth=0.4)
    plt.plot(w_1_freq[:-1], w_1_ps[:-1] / ((Ii[:, 2][node_test_S_a] * U_bar[node_test_S_a]) ** 2) * w_1_freq[:-1], '--', color='brown',alpha=0.3, linewidth=0.4)
    plt.scatter(u_1_freq[:-1], u_1_ps[:-1] / ((Ii[:, 0][node_test_S_a] * U_bar[node_test_S_a]) ** 2) * u_1_freq[:-1], marker='o',edgecolors='none', color='blue',label='Measur. u', alpha=0.6, s=6)
    plt.scatter(v_1_freq[:-1], v_1_ps[:-1] / ((Ii[:, 1][node_test_S_a] * U_bar[node_test_S_a]) ** 2) * v_1_freq[:-1], marker='>',edgecolors='none', color='orange', label='Measur. v', alpha=0.6, s=6)
    plt.scatter(w_1_freq[:-1], w_1_ps[:-1] / ((Ii[:, 2][node_test_S_a] * U_bar[node_test_S_a]) ** 2) * w_1_freq[:-1], marker='v',edgecolors='none', color='brown', label='generated w', alpha=0.6, s=6)
    plt.xscale('log')
    # plt.xlim([0.001,2])
    plt.xlabel('Frequency [Hz]')
    plt.grid(which='both')
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.savefig(rf"{export_folder}\non-dim-auto-spectrum_standard_VS_measurements_at_node_" + str(node_test_S_a))
    plt.close()
    from matplotlib.legend_handler import HandlerTuple
    plt.figure(dpi=500)
    plt.axis('off')
    plt.legend(handles[:3]+[tuple(handles[3:])], labels[:3] + ['Generated'], handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=4, markerscale=1.8)
    plt.savefig(rf"{export_folder}\legend_non-dim-auto-spectrum_standard_VS_measurements_at_node_" + str(node_test_S_a))
    plt.tight_layout()
    plt.close()

    # =============================================================================
    # Plotting auto-spectra - Dimensional
    # =============================================================================
    # wind_block_T = min(600, wind_T)  # s. Duration of each segment, to build an average in the Welch method
    nperseg = len(windspeed_u[node_test_S_a]) / round(wind_T / wind_block_T)
    u_1_freq, u_1_ps = signal.welch(windspeed_u[node_test_S_a], wind_freq, nperseg=nperseg)
    v_1_freq, v_1_ps = signal.welch(windspeed_v[node_test_S_a], wind_freq, nperseg=nperseg)
    w_1_freq, w_1_ps = signal.welch(windspeed_w[node_test_S_a], wind_freq, nperseg=nperseg)
    plt.figure(figsize=(15, 8))
    plt.title('Auto-Spectrum S(f)')  # Dimensional f
    plt.plot(f_array, S_a_nondim[:, node_test_S_a, 0] * ((Ii[:, 0][node_test_S_a] * U_bar[node_test_S_a]) ** 2) / f_array,
             color='blue', label='target u')
    plt.plot(f_array, S_a_nondim[:, node_test_S_a, 1] * ((Ii[:, 1][node_test_S_a] * U_bar[node_test_S_a]) ** 2) / f_array,
             color='orange', label='target v')
    plt.plot(f_array, S_a_nondim[:, node_test_S_a, 2] * ((Ii[:, 2][node_test_S_a] * U_bar[node_test_S_a]) ** 2) / f_array,
             color='green', label='target w')
    plt.plot(u_1_freq, u_1_ps, '--', color='blue', alpha=0.4, linewidth=0.5)
    plt.plot(v_1_freq, v_1_ps, '--', color='orange', alpha=0.4, linewidth=0.5)
    plt.plot(w_1_freq, w_1_ps, '--', color='green', alpha=0.4, linewidth=0.5)
    plt.scatter(u_1_freq, u_1_ps, color='blue', label='generated u', alpha=0.6, s=2)
    plt.scatter(v_1_freq, v_1_ps, color='orange', label='generated v', alpha=0.6, s=2)
    plt.scatter(w_1_freq, w_1_ps, color='green', label='generated w', alpha=0.6, s=2)
    plt.xscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.xlim([1 / wind_T / 1.2, wind_freq / 2 * 1.2])  # 1.2 zoom-out ratio
    plt.legend()
    plt.grid(which='both')
    plt.savefig(rf"{export_folder}\dim-auto-spectrum_standard_VS_measurements_at_node_" + str(node_test_S_a))
    plt.close()

    # =============================================================================
    # Plotting coherence, function of freq and spatial separation
    # =============================================================================
    from wind_field.coherence import coherence

    fig = plt.figure(figsize=(6 * n_nodes_val_coh, 6 * n_nodes_val_coh), dpi=100)
    nperseg = 512  # Welch's method. Length of each segment.
    counter = 0
    for node_1 in range(n_nodes_val_coh):
        for node_2 in range(n_nodes_val_coh):
            counter += 1

            if node_2 > node_1:
                ax = fig.add_subplot(n_nodes_val_coh, n_nodes_val_coh, counter)
                coh_freq_u = coherence(windspeed_u[node_1], windspeed_u[node_2], fs=wind_freq, nperseg=nperseg)['freq']
                coh_u = coherence(windspeed_u[node_1], windspeed_u[node_2], fs=wind_freq, nperseg=nperseg)['cocoh']
                ax.set_title('Node ' + str(node_1) + ' and ' + str(node_2))
                plt.plot(f_array, np.e ** (-np.sqrt((Cij[0, 0] * delta_x[node_1, node_2]) ** 2 +
                                                 (Cij[0, 1] * delta_y[node_1, node_2]) ** 2 +
                                                 (Cij[0, 2] * delta_z[node_1, node_2]) ** 2) *
                                        f_array / ((U_bar[node_1] + U_bar[node_2]) / 2)), color='blue', label='target u',
                         alpha=0.75, linewidth=3)
                plt.scatter(coh_freq_u, coh_u, alpha=0.6, s=4, color='blue', label='generated u')

                coh_freq_v = coherence(windspeed_v[node_1], windspeed_v[node_2], fs=wind_freq, nperseg=nperseg)['freq']
                coh_v = coherence(windspeed_v[node_1], windspeed_v[node_2], fs=wind_freq, nperseg=nperseg)['cocoh']
                plt.plot(f_array, np.e ** (-np.sqrt((Cij[1, 0] * delta_x[node_1, node_2]) ** 2 +
                                                 (Cij[1, 1] * delta_y[node_1, node_2]) ** 2 +
                                                 (Cij[1, 2] * delta_z[node_1, node_2]) ** 2) *
                                        f_array / ((U_bar[node_1] + U_bar[node_2]) / 2)), color='orange', label='target v',
                         alpha=0.5, linestyle='dashed', linewidth=3)
                plt.scatter(coh_freq_v, coh_v, alpha=0.6, s=4, color='orange', label='generated v')

                coh_freq_w = coherence(windspeed_w[node_1], windspeed_w[node_2], fs=wind_freq, nperseg=nperseg)['freq']
                coh_w = coherence(windspeed_w[node_1], windspeed_w[node_2], fs=wind_freq, nperseg=nperseg)['cocoh']
                plt.plot(f_array, np.e ** (-np.sqrt((Cij[2, 0] * delta_x[node_1, node_2]) ** 2 +
                                                 (Cij[2, 1] * delta_y[node_1, node_2]) ** 2 +
                                                 (Cij[2, 2] * delta_z[node_1, node_2]) ** 2) *
                                        f_array / ((U_bar[node_1] + U_bar[node_2]) / 2)), color='green', label='target w',
                         alpha=0.5, linestyle='dotted', linewidth=3)
                plt.scatter(coh_freq_w, coh_w, alpha=0.6, s=4, color='green', label='generated w')
                plt.xlim([-0.05, 2])
                plt.xlabel('freq [Hz]')
                plt.ylabel('Coherence')
                plt.grid()
                plt.legend()
    plt.tight_layout()
    plt.savefig(rf'{export_folder}\co-coherence')
    plt.close()

    # =============================================================================
    # Plotting correlation coefficients between g_nodes (function of spatial separation in x,y and z)
    # =============================================================================
    corrcoef_u_target = np.zeros((n_nodes_validated, n_nodes_validated))
    corrcoef_v_target = np.zeros((n_nodes_validated, n_nodes_validated))
    corrcoef_w_target = np.zeros((n_nodes_validated, n_nodes_validated))
    for n in range(n_nodes_validated):
        for m in range(n_nodes_validated):
            corrcoef_u_target[n, m] = np.exp(-np.sqrt((delta_x[n, m] / iLj_avg[m, n, 0, 0]) ** 2 + (delta_y[n, m] / iLj_avg[m, n, 0, 1]) ** 2 + (delta_z[n, m] / iLj_avg[m, n, 0, 2]) ** 2))
            corrcoef_v_target[n, m] = np.exp(-np.sqrt((delta_x[n, m] / iLj_avg[m, n, 1, 0]) ** 2 + (delta_y[n, m] / iLj_avg[m, n, 1, 1]) ** 2 + (delta_z[n, m] / iLj_avg[m, n, 1, 2]) ** 2))
            corrcoef_w_target[n, m] = np.exp(-np.sqrt((delta_x[n, m] / iLj_avg[m, n, 2, 0]) ** 2 + (delta_y[n, m] / iLj_avg[m, n, 2, 1]) ** 2 + (delta_z[n, m] / iLj_avg[m, n, 2, 2]) ** 2))
    corrcoef_u = np.corrcoef(windspeed_u[:n_nodes_validated])
    corrcoef_v = np.corrcoef(windspeed_v[:n_nodes_validated])
    corrcoef_w = np.corrcoef(windspeed_w[:n_nodes_validated])
    corrcoef = np.array([corrcoef_u, corrcoef_v, corrcoef_w])
    corrcoef_target = np.array([corrcoef_u_target, corrcoef_v_target, corrcoef_w_target])
    from mpl_toolkits.mplot3d import Axes3D
    del Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    X, Y = np.meshgrid(list(range(n_nodes_validated)), list(range(n_nodes_validated)))
    colors = [[], [], []]
    for i in range(3):
        colors[i] = cm.rainbow(abs(corrcoef[i] - corrcoef_target[i]))  ##### corrcoef[i]-corrcoef_target[i]
    fig = plt.figure(figsize=(16, 6))
    plt.suptitle('Top: Correlation coefficients \n Bottom: Corr. coeff. difference between expected and generated')
    for i in range(3):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax = fig.gca(projection='3d')
        ax.set_title(str(['u', 'v', 'w'][i]), fontweight='bold')
        surf1 = ax.plot_surface(X, Y, corrcoef[i], linewidth=0.2, antialiased=True, color='orange', alpha=0.9,
                                label='generated')
        ax.plot_surface(X, Y, corrcoef_target[i], linewidth=0.2, antialiased=True, color='green', alpha=0.5,
                        label='expected')
        ax.set_xlabel('Node Number')
        ax.set_ylabel('Node Number')
        ax.set_zlabel('corr. coeff.')
        ax.view_init(elev=20, azim=240)
        ax.text(n_nodes_validated, 0, 0.15, "generated", color='orange')
        ax.text(n_nodes_validated, 0, 0, "expected", color='green')
    for i in range(3):
        ax = fig.add_subplot(2, 3, i + 4)
        ax = fig.gca()
        img = ax.imshow(colors[i], cmap=cm.rainbow)
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(img)
        ax.set_xlabel('Node Number')
        ax.set_ylabel('Node Number')
        cbar.set_label('$\Delta$ corr. coeff.')
    plt.savefig(rf'{export_folder}\correlation_coefficients')
    plt.close()

    # =============================================================================
    # Plotting normalized co-spectrum validation by comparing eq.(3.40) with eq.(3.41) in Strømmen's book
    # =============================================================================
    nperseg = int(len(windspeed_u[node_test_S_a]) / 20)
    n_freqs_plotted = 217  # manually input this
    # Measured values:
    freq_csd = signal.csd(windspeed_u[0], windspeed_u[0], wind_freq, window='hann', nperseg=nperseg)[0]
    csd_u = np.zeros((n_nodes_validated, n_nodes_validated, len(freq_csd)))
    csd_v = np.zeros((n_nodes_validated, n_nodes_validated, len(freq_csd)))
    csd_w = np.zeros((n_nodes_validated, n_nodes_validated, len(freq_csd)))
    cospec_norm_u = np.zeros((n_nodes_validated, n_nodes_validated, len(freq_csd)))
    cospec_norm_v = np.zeros((n_nodes_validated, n_nodes_validated, len(freq_csd)))
    cospec_norm_w = np.zeros((n_nodes_validated, n_nodes_validated, len(freq_csd)))

    for n in range(n_nodes_validated):
        for m in range(n_nodes_validated):
            csd_u[n,m,:] = np.real(signal.csd(windspeed_u[n], windspeed_u[m], wind_freq, window='hann', nperseg=nperseg)[1])
            csd_v[n,m,:] = np.real(signal.csd(windspeed_v[n], windspeed_v[m], wind_freq, window='hann', nperseg=nperseg)[1])
            csd_w[n,m,:] = np.real(signal.csd(windspeed_w[n], windspeed_w[m], wind_freq, window='hann', nperseg=nperseg)[1])
    for n in range(n_nodes_validated):
        for m in range(n_nodes_validated):
            cospec_norm_u[n,m,:] = csd_u[n,m,:] / np.sqrt(np.multiply(csd_u[n,n,:], csd_u[m,m,:]))
            cospec_norm_v[n,m,:] = csd_v[n,m,:] / np.sqrt(np.multiply(csd_v[n,n,:], csd_v[m,m,:]))
            cospec_norm_w[n,m,:] = csd_w[n,m,:] / np.sqrt(np.multiply(csd_w[n,n,:], csd_w[m,m,:]))

    # Expected values
    cospec_norm_u_target = np.e**(-np.einsum('f,mn->mnf', freq_csd , np.sqrt( (Cij[0,0]*delta_x)**2 + (Cij[0,1]*delta_y)**2 + (Cij[0,2]*delta_z)**2 )  /  U_bar_avg))
    cospec_norm_v_target = np.e**(-np.einsum('f,mn->mnf', freq_csd , np.sqrt( (Cij[1,0]*delta_x)**2 + (Cij[1,1]*delta_y)**2 + (Cij[1,2]*delta_z)**2 )  /  U_bar_avg))
    cospec_norm_w_target = np.e**(-np.einsum('f,mn->mnf', freq_csd , np.sqrt( (Cij[2,0]*delta_x)**2 + (Cij[2,1]*delta_y)**2 + (Cij[2,2]*delta_z)**2 )  /  U_bar_avg))

    # Plotting for delta_y
    from matplotlib.pyplot import cm
    from matplotlib.lines import Line2D
    from cycler import cycler

    fig = plt.figure(figsize=(7,5), dpi=400)
    ax = fig.gca(projection='3d')
    plt.title(r'$Re(S_{uu}(f,\Delta_{X_u}=0,\Delta_{Y_v},\Delta_{Z_w}=0))\/ / \/S_u(f)$')
    # color = cm.tab10(np.linspace(0,1,len(delta_y[0])))
    color = cm.plasma(np.linspace(0., 0.9, len(delta_y[0])))
    for i in range(n_nodes_validated):
        plt.plot( [delta_y[0][i]] * len(freq_csd[:n_freqs_plotted]), freq_csd[:n_freqs_plotted], cospec_norm_u[0][i][:n_freqs_plotted], marker='o', color=color[i], linestyle='--', linewidth=0.5, alpha=0.7, markersize=2)
        plt.plot( [delta_y[0][i]] * len(freq_csd[:n_freqs_plotted]), freq_csd[:n_freqs_plotted], cospec_norm_u_target[0][i][:n_freqs_plotted], color=color[i])
    ax.set_ylabel('Frequency [Hz]')
    # ax.set_ylim3d(0, freq_csd[-1])
    ax.set_ylim3d(0, 0.4)
    ax.set_xlabel('Spacing $\Delta_{Y_v}$ [m]')
    ax.set_xlim3d(delta_y[0][0], delta_y[0][-1])
    ax.set_zlim3d(0, 1)
    ax.view_init(30,60)
    plt.tight_layout()
    legend_lines = [Line2D([0], [0], color='black'),Line2D([0], [0], color='black', marker='o', linestyle='--', alpha=0.5)]
    plt.savefig(rf'{export_folder}\normalized_co-spectrum.pdf')
    plt.close()
    plt.figure(dpi=400)
    plt.axis('off')
    plt.legend(legend_lines, ['Target','Generated'], ncol=2)
    # plt.show()
    plt.savefig(rf'{export_folder}\legend_normalized_co-spectrum')
    plt.close()
    return None
#
# ## Validation:
# from buffeting import deg, rad, U_bar_func, Ai_func, Cij_func, Ii_func, iLj_func, beta_0_func
# from straight_bridge_geometry import g_node_coor, arc_length, R
# from wind_field.wind_field_3D import wind_field_3D_func
# from transformations import T_GsGw_func
# import numpy as np
#
# # Input parameters
# beta_DB = rad(100)
# theta_G = rad(0)
# T = 1200  # time domain
# dt = 0.5  # time domain. Best: 0.1 s when plotting auto-spectrum
# n_freq = 2048
# f_min = 0.005
# f_max = 6
# # g_node_coor = g_node_coor[:10]
# # Alternative manual coordinates
# n_nodes_validated = 11
# node_spacing = 5  # meters
# g_node_coor = np.transpose(np.array([np.arange(n_nodes_validated)*5, [0]*n_nodes_validated, [14.5]*n_nodes_validated]))
#
# # Other parameters
# beta_G = beta_0_func(beta_DB=beta_DB)
# T_GwGs = np.transpose(T_GsGw_func(beta_0=beta_G, theta_0=theta_G))
# g_node_coor_Gw = np.einsum('ij,nj->ni' , T_GwGs, g_node_coor)
# U_bar = U_bar_func(g_node_coor)
# Ai = Ai_func(cond_rand_A=False)
# Cij = Cij_func(cond_rand_C=False).flatten()  # [Cux,Cuy,Cuz,Cvx,Cvy,Cvz,Cwx,Cwy,Cwz]
# Ii = Ii_func(g_node_coor, beta_DB=False, Ii_simplified=True)
# iLj_bad_shape = iLj_func(g_node_coor)  # first node defines iLj (homogeneity required)
# iLj = np.transpose(np.array([i.flatten() for i in iLj_bad_shape]))
# wind_field = wind_field_3D_func(node_coor_wind=g_node_coor_Gw, V=U_bar, Ai=Ai, Cij=Cij, I=Ii, iLj=iLj, T=T, sample_freq=1/dt, spectrum_type=2)
# windspeed = wind_field['windspeed']
#
#
# wind_field_3D_applied_validation_func(g_node_coor=g_node_coor, windspeed=windspeed, dt=dt, wind_block_T=T, beta_DB=beta_DB, arc_length=arc_length, R=R,
#                                       Ii_simplified_bool=True, f_min=f_min, f_max=f_max, n_freq=n_freq, n_nodes_validated=n_nodes_validated, node_test_S_a=0, n_nodes_val_coh=3)



