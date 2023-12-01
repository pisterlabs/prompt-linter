import re
from pathlib import Path

import numpy as np

from .. import io, polsar, utils
from ..io.project import stand_pol_rat_files
# from timer import Timer
from ..opertaions import coherence, phase_diff, power_ratio
from ..stats.timer import Timer


class StandardPolarimetric:
    def __init__(
        self,
        DIR_campaign,
        ID_campaign,
        ID_flight,
        band,
        ID_Pass,
        n_try="01",
        save_path=None,
        n_windows=(7, 7),
        coh=None,
        crop=None,
        crop_az=None,
        crop_rg=None,
        *args,
        **kwargs,
    ):

        master_files = stand_pol_rat_files(
            DIR_campaign, ID_campaign, ID_flight, band, ID_Pass, n_try="01"
        )
        pat = r"incidence_(.*?)_t.*"
        patch_name = re.findall(pat, Path(master_files["AOI"]).name)[0]

        with Timer(name="read_files"):
            fscl = np.sqrt(np.sin(io.rat.loadrat(master_files["AOI"])))
            if crop is None:
                self.slc_hh = io.rat.loadrat(master_files["HH"]) * fscl
                self.slc_hv = io.rat.loadrat(master_files["HV"]) * fscl
                self.slc_vh = io.rat.loadrat(master_files["VH"]) * fscl
                self.slc_vv = io.rat.loadrat(master_files["VV"]) * fscl

            else:
                self.slc_hh = (io.rat.loadrat(master_files["HH"]) * fscl)[
                    crop_rg[0] : crop_rg[1], crop_az[0] : crop_az[1]
                ]
                self.slc_hv = (io.rat.loadrat(master_files["HV"]) * fscl)[
                    crop_rg[0] : crop_rg[1], crop_az[0] : crop_az[1]
                ]
                self.slc_vh = (io.rat.loadrat(master_files["VH"]) * fscl)[
                    crop_rg[0] : crop_rg[1], crop_az[0] : crop_az[1]
                ]
                self.slc_vv = (io.rat.loadrat(master_files["VV"]) * fscl)[
                    crop_rg[0] : crop_rg[1], crop_az[0] : crop_az[1]
                ]

        print("Patch Name\t: {}".format(patch_name))
        print(
            "Range\t\t: {}\nAzimuth\t\t: {}".format(
                self.slc_hh.shape[0], self.slc_hh.shape[1]
            )
        )

        with Timer(name="scat_matrix"):
            # Do something
            scat_matrix = self.cal_scattering_matrix(
                self.slc_hh, self.slc_hv, self.slc_vh, self.slc_vv
            )

        with Timer(name="slc_pauli"):
            # Do something
            self.slc_pauli = polsar.decomposition.operators.pauli_vec(
                scat_matrix
            )

        with Timer(name="polarimetric_matrix_jit"):
            # Do something
            self.slc_t44 = polsar.decomposition.operators.polarimetric_matrix_jit(
                self.slc_pauli
            )

        with Timer(name="t_pauli_t_mat"):
            # Do something
            self.pauli_t_mat = np.stack(
                (
                    self.slc_t44[:, :, 0, 0],
                    self.slc_t44[:, :, 1, 1],
                    self.slc_t44[:, :, 2, 2],
                ),
                axis=2,
            )

        with Timer(name="eigen_decomposition_jit"):
            # Do something
            (
                self.slc_ew4,
                self.slc_ev44,
            ) = polsar.decomposition.eigen.eigen_decomposition_jit(self.slc_t44)

        with Timer(name="ent_ani_alp_nc"):
            # Do something
            self.slc_ent_ani_alp_44_nc = polsar.parameters.ent_ani_alp(
                self.slc_ew4[:, :, 1:] - (self.slc_ew4[:, :, 0:1]),
                self.slc_ev44[:, :, 0:1, 1:4],
            )

        if coh is not None:
            with Timer(name="coh"):
                self.coh_hhvv = coherence(
                    self.slc_hh, self.slc_vv, window_size=7
                )
                self.coh_snr = coherence(
                    self.slc_hv, self.slc_vh, window_size=7
                )
                self.coh_hhhv = coherence(
                    self.slc_hh, self.slc_hv, window_size=7
                )
                self.coh_vvvh = coherence(
                    self.slc_vv, self.slc_hv, window_size=7
                )
                self.coh_hhxx = coherence(
                    self.slc_hh, (self.slc_vh + self.slc_vh) / 2, window_size=7
                )
                self.coh_vvxx = coherence(
                    self.slc_vv, (self.slc_vh + self.slc_vh) / 2, window_size=7
                )

        with Timer(name="power_phase_diff"):
            # HH-VV Phasedifference
            self.ph_diff_hhvv = phase_diff(
                self.slc_hh, self.slc_vv, window_size=7, deg=True
            )
            self.p_hhvv = power_ratio(self.slc_hh, self.slc_vv, window_size=7)

        with Timer(name="write_h5f"):
            # Do something

            import h5py

            if save_path is not None:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                h5_filename = Path.joinpath(
                    Path(save_path), str(patch_name + "_processed.h5")
                )
                h5f = h5py.File(str(h5_filename), "w")
                h5f.create_dataset("pauli_t_mat", data=self.pauli_t_mat)
                # h5f.create_dataset('mlc_ew4', data=self.slc_ew4)
                # h5f.create_dataset('mlc_ev44', data=self.slc_ev44)
                h5f.create_dataset(
                    "entropy", data=self.slc_ent_ani_alp_44_nc[0, :, :]
                )
                h5f.create_dataset(
                    "anisotropy", data=self.slc_ent_ani_alp_44_nc[1, :, :]
                )
                h5f.create_dataset(
                    "alpha",
                    data=np.rad2deg(self.slc_ent_ani_alp_44_nc[2, :, :]),
                )
                h5f.create_dataset("ph_diff_hhvv", data=self.ph_diff_hhvv)
                h5f.create_dataset("p_hhvv", data=self.p_hhvv)

                if coh is not None:
                    h5f.create_dataset("coh_hhvv", data=self.coh_hhvv)
                    h5f.create_dataset("coh_snr", data=self.coh_snr)
                    h5f.create_dataset("coh_hhhv", data=self.coh_hhhv)
                    h5f.create_dataset("coh_vvvh", data=self.coh_vvvh)
                    h5f.create_dataset("coh_hhxx", data=self.coh_hhxx)
                    h5f.create_dataset("coh_vvxx", data=self.coh_vvxx)
                h5f.close()

    def cal_scattering_matrix(self, slc_hh, slc_hv, slc_vh, slc_vv):
        """
        """
        scat_matrix = np.zeros(
            (slc_hh.shape[0], slc_hh.shape[1], 4), dtype=np.complex_
        )
        scat_matrix[:, :, 0] = slc_hh
        scat_matrix[:, :, 1] = slc_hv
        scat_matrix[:, :, 2] = slc_vh
        scat_matrix[:, :, 3] = slc_vv

        return scat_matrix


# Processing Class for Standard Polarimetric

# (Each Coherence Combination, Entropy, Anisotropy, Alpha)
