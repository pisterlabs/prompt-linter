# -*- coding: utf-8 -*-
# ***********************************************************************
# ******************  CANADIAN ASTRONOMY DATA CENTRE  *******************
# *************  CENTRE CANADIEN DE DONNÉES ASTRONOMIQUES  **************
#
#  (c) 2020.                            (c) 2020.
#  Government of Canada                 Gouvernement du Canada
#  National Research Council            Conseil national de recherches
#  Ottawa, Canada, K1A 0R6              Ottawa, Canada, K1A 0R6
#  All rights reserved                  Tous droits réservés
#
#  NRC disclaims any warranties,        Le CNRC dénie toute garantie
#  expressed, implied, or               énoncée, implicite ou légale,
#  statutory, of any kind with          de quelque nature que ce
#  respect to the software,             soit, concernant le logiciel,
#  including without limitation         y compris sans restriction
#  any warranty of merchantability      toute garantie de valeur
#  or fitness for a particular          marchande ou de pertinence
#  purpose. NRC shall not be            pour un usage particulier.
#  liable in any event for any          Le CNRC ne pourra en aucun cas
#  damages, whether direct or           être tenu responsable de tout
#  indirect, special or general,        dommage, direct ou indirect,
#  consequential or incidental,         particulier ou général,
#  arising from the use of the          accessoire ou fortuit, résultant
#  software.  Neither the name          de l'utilisation du logiciel. Ni
#  of the National Research             le nom du Conseil National de
#  Council of Canada nor the            Recherches du Canada ni les noms
#  names of its contributors may        de ses  participants ne peuvent
#  be used to endorse or promote        être utilisés pour approuver ou
#  products derived from this           promouvoir les produits dérivés
#  software without specific prior      de ce logiciel sans autorisation
#  written permission.                  préalable et particulière
#                                       par écrit.
#
#  This file is part of the             Ce fichier fait partie du projet
#  OpenCADC project.                    OpenCADC.
#
#  OpenCADC is free software:           OpenCADC est un logiciel libre ;
#  you can redistribute it and/or       vous pouvez le redistribuer ou le
#  modify it under the terms of         modifier suivant les termes de
#  the GNU Affero General Public        la “GNU Affero General Public
#  License as published by the          License” telle que publiée
#  Free Software Foundation,            par la Free Software Foundation
#  either version 3 of the              : soit la version 3 de cette
#  License, or (at your option)         licence, soit (à votre gré)
#  any later version.                   toute version ultérieure.
#
#  OpenCADC is distributed in the       OpenCADC est distribué
#  hope that it will be useful,         dans l’espoir qu’il vous
#  but WITHOUT ANY WARRANTY;            sera utile, mais SANS AUCUNE
#  without even the implied             GARANTIE : sans même la garantie
#  warranty of MERCHANTABILITY          implicite de COMMERCIALISABILITÉ
#  or FITNESS FOR A PARTICULAR          ni d’ADÉQUATION À UN OBJECTIF
#  PURPOSE.  See the GNU Affero         PARTICULIER. Consultez la Licence
#  General Public License for           Générale Publique GNU Affero
#  more details.                        pour plus de détails.
#
#  You should have received             Vous devriez avoir reçu une
#  a copy of the GNU Affero             copie de la Licence Générale
#  General Public License along         Publique GNU Affero avec
#  with OpenCADC.  If not, see          OpenCADC ; si ce n’est
#  <http://www.gnu.org/licenses/>.      pas le cas, consultez :
#                                       <http://www.gnu.org/licenses/>.
#
#  $Revision: 4 $
#
# ***********************************************************************
#

import aplpy
import h5py
import os

import matplotlib as mpl
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from astropy.table import Table
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from caom2 import ProductType, ReleaseType, ObservationIntentType
from caom2pipe import astro_composable as ac
from caom2pipe import manage_composable as mc
from cfht2caom2 import metadata as md

__all__ = ['visit']


class CFHTPreview(mc.PreviewVisitor):
    def __init__(self, instrument, intent, obs_type, target, **kwargs):
        super(CFHTPreview, self).__init__(**kwargs)
        self._instrument = md.Inst(instrument)
        self._intent = intent
        self._obs_type = obs_type
        self._preview_fqn = os.path.join(
            self._working_dir, self._storage_name.prev
        )
        self._thumb_fqn = os.path.join(
            self._working_dir, self._storage_name.thumb
        )
        self._zoom_fqn = os.path.join(
            self._working_dir, self._storage_name.zoom
        )
        self._target_name = (
            target.name if target is not None else self._storage_name.file_id
        )

    def generate_plots(self, obs_id):
        self._logger.debug(f'Begin generate_plots for {obs_id}')
        if self._instrument is md.Inst.SITELLE and (self._storage_name.suffix == 'p' or self._storage_name.hdf5):
            if self._storage_name.suffix == 'p':
                count = self._sitelle_calibrated_cube()
            else:
                count = self._sitelle_hdf5()
        elif (
            self._instrument is md.Inst.ESPADONS
            and self._storage_name.suffix in ['i', 'p']
        ):
            count = self._do_espadons_science()
        elif (
            self._instrument is md.Inst.SPIROU
            and self._storage_name.suffix in ['e', 'p', 's', 't', 'v']
        ):
            if self._storage_name.suffix == 'v':
                count = self._do_spirou_bintable()
            else:
                count = self._do_spirou_intensity_spectrum()
        else:
            count = 0
            if not self._storage_name.hdf5 and not '_diag' in self._storage_name.file_name:
                count = self._do_ds9_prev(obs_id)
        self._logger.debug('End generate_plots')
        return count

    def _do_espadons_science(self):
        self._logger.debug(
            f'Do espadons science preview augmentation with '
            f'{self._science_fqn}'
        )
        # from genEspaprevperplane.py

        # Polarization scale factor
        pScale = 5.0

        self._hdu_list = fits.open(self._science_fqn)
        self._ext = 0
        try:
            ignore = self._hdu_list[self._ext].header.get('OBJECT')
        except LookupError:
            self._ext = 1
            ignore = self._hdu_list[self._ext].header.get('OBJECT')

        hdr = self._hdu_list[self._ext].header.copy()
        bzero = hdr.get('BZERO')
        bscale = hdr.get('BSCALE')

        if bzero is not None and bzero > 0.0:
            data_0 = self._hdu_list[self._ext].data[0].copy()
            data_1 = self._hdu_list[self._ext].data[1].copy()
            # wavelength array (nm)
            sw = bscale * data_0 - bzero
            # intensity array (normalized)
            si = bscale * data_1 - bzero
            if self._storage_name.suffix == 'p':
                # Stokes array
                data_2 = self._hdu_list[self._ext].data[2].copy()
                sp = bscale * data_2 - bzero
        else:
            sw = self._hdu_list[self._ext].data[0].copy()  # wavelength array (nm)
            si = self._hdu_list[self._ext].data[1].copy()  # intensity array (normalized)
            if self._storage_name.suffix == 'p':
                sp = self._hdu_list[self._ext].data[2].copy()  # Stokes array

        self._logger.debug(f'{sw.shape} {sw}, {si}')
        npix = sw.shape[0]

        swa = 10.0 * sw
        sia = np.arange(0.0, npix, 1.0)
        spa = None
        if self._storage_name.suffix == 'p':
            spa = np.arange(0.0, npix, 1.0)

        # determine upper/lower y limits for two planes from intensity values
        for i in range(sia.shape[0]):
            sia[i] = float(si[i])
            if self._storage_name.suffix == 'p':
                spa[i] = (
                    float(sp[i]) * pScale
                )  # increase scale of polarization

        fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
        count = 0
        if self._subplot(fig, swa, sia, spa, 4300.0, 4600.0, 1, 4408.0, 4412.0, 'Stokes spectrum (x5)'):
            if self._subplot(fig, swa, sia, spa, 6500.0, 6750.0, 2, 6589.0, 6593.0, 'Stokes spectrum (x5)'):
                plt.savefig(self._preview_fqn, format='jpg')
                self.add_preview(
                    self._storage_name.prev_uri, self._storage_name.prev, ProductType.PREVIEW, ReleaseType.DATA
                )
                self.add_to_delete(self._preview_fqn)
                count = 1
                count += self._gen_thumbnail()
                if count == 2:
                    self.add_preview(
                        self._storage_name.thumb_uri, self._storage_name.thumb, ProductType.THUMBNAIL, ReleaseType.META
                    )
                    self.add_to_delete(self._thumb_fqn)
        return count

    def _subplot(
        self,
        fig,
        swa,
        sia,
        spa,
        wl_low,
        wl_high,
        subplot_index,
        text_1,
        text_2,
        text_3,
    ):
        label = f'{self._storage_name.product_id}: {self._target_name}'
        wl = swa[(swa > wl_low) & (swa < wl_high)]
        flux = sia[(swa > wl_low) & (swa < wl_high)]
        wl_sort = wl[wl.argsort()]
        flux_sort = flux[wl.argsort()]
        if self._storage_name.suffix == 'p':
            pflux = spa[(swa > wl_low) & (swa < wl_high)]
            pflux_sort = pflux[wl.argsort()]
            flux = np.append(flux, pflux)
        result = False
        if flux.shape == (0,):
            self._logger.warning(f'Sub-plot flux ill-defined for {self._storage_name.file_name}')
        else:
            ymax = 1.1 * np.max(flux)
            if np.isnan(ymax):
                ymax = 1.0
            ymin = np.min([0.0, np.min(flux) - (ymax - np.max(flux))])
            if np.isnan(ymin):
                ymin = 0.0

            # pylab.subplot(2, 1, subplot_index)
            axis = fig.add_subplot(2, 1, subplot_index)
            axis.grid(True)
            axis.plot(wl_sort, flux_sort, color='k')
            if self._storage_name.suffix == 'p':
                axis.plot(wl_sort, pflux_sort, color='b')
                axis.text(
                    text_1,
                    (ymin + 0.02 * (ymax - ymin)),
                    text_3,
                    size=16,
                    color='b',
                )
            axis.set(
                title=label,
                xlabel=r'Wavelength ($\AA$)',
                ylabel='Relative Intensity',
            )
            axis.title.set_weight('bold')
            axis.title.set_color('m')
            axis.text(
                text_2,
                (ymin + 0.935 * (ymax - ymin)),
                'Intensity spectrum',
                size=16,
            )
            axis.set_ylim(ymin, ymax)
            result = True
        return result

    def _do_ds9_prev(self, obs_id):
        """
                    256               1024                zoom
        ESPaDOnS:
        mosaic      ''                ''                  -fits
        pan         ''                ''                  ''
        rotate      ''                ''                  ''
        scale       zscale            zscale              zscale
        scope       global            global              global
        mode        -mode none        -mode none          -mode none
        zoom        to fit            to fit              1

        MegaPrime, not 'p' and 'o':
        mosaic     -mosaicimage iraf -mosaicimage iraf    -fits
        pan        ''                ''                   -pan -9 1780
        scale      zscale            zscale               zscale
        scope      local             local                global
        mode       -mode none        ''                   -mode none
        zoom       to fit            to fit               1

        MegaPrime, 'p' and 'o':
        mosaic     -mosaicimage wcs  -mosaicimage wcs     -fits
        pan        ''                ''                   -pan -9 1780
        scale      zscale            zscale               zscale
        scope      global            global               global
        mode       -mode none        ''                   -mode none
        zoom       to fit            to fit               1

        MegaPrime extensions:
        rotate[23] -rotate 180       -rotate 180          ''
        rotate[14] -rotate 180       -rotate 180          -rotate 180
        rotate[1]  -rotate 180       -rotate 180          -rotate 180

        WIRCam 'o', 'p', 'and 's':
        mosaic     -mosaicimage wcs  -mosaicimage wcs     -fits
        rotate     ''                ''                   ''
        scale      zscale            zscale               zscale
        scope      global            global               global
        mode       -mode none        ''                   -mode none
        zoom       to fit            to fit               1

        WIRCam not 'o', 'p', 'and 's':
        mosaic     -mosaicimage iraf -mosaicimage iraf    -fits
        rotate     ''                ''                   ''
        scale      zscale            zscale               zscale
        scope      local             local                global
        mode       -mode none        ''                   -mode none
        zoom       to fit            to fit               1

        WIRCam extensions:
        pan[4]     ''               ''                    -pan 484 -484
        pan[1]     ''               ''                    -pan -484 -484

        SITELLE 2D images:
        mosaic     ''               ''                    -fits
        pan        ''               ''                    -pan -512 1544
        rotate     ''               ''                    ''
        scale      zscale           zscale                zscale
        scope      global           global                global
        mode       -mode none       -mode none            -mode none
        zoom       to fit           to fit                1

        SPIRou Raw 2D:
        mosaic     ''               ''                    -fits
        pan        ''               ''                    ''
        rotate     ''               ''                    ''
        scale      zscale           zscale                zscale
        scope      global           global                global
        mode       -mode none       -mode none            -mode none
        zoom       to fit           to fit                1

        """
        self._logger.debug(
            f'Do ds9 preview augmentation with {self._science_fqn}'
        )
        count = 0
        delete_list = []
        content = ac.read_fits_data(self._science_fqn)
        headers = [h.header for h in content]
        # SF - 26-02-21
        # use the size of the HDUList
        num_extensions = len(content)

        zoom_science_fqn = self._science_fqn

        # from genWirprevperplane.py
        # if it's a datacube, just take the first slice
        # e.g. fitscopy '928690p.fits[*][*,*,1:1]' s1928690p.fits

        # set up the correct input file - may need to use fitscopy
        rotate_param = ''
        scale_param = 'zscale'
        if self._instrument is md.Inst.WIRCAM:
            if self._science_fqn.endswith('.fz'):
                naxis_3 = headers[0].get('ZNAXIS3', 1)
            else:
                naxis_3 = headers[0].get('NAXIS3', 1)

            # SF - 08-04-20 - for 'g' use fitscopy, then regular ds9 for zoom
            # calibration. This is a change from guidance of 19-03-20, which
            # was to use the previews from 'p' files for the 'g' files
            if naxis_3 != 1 or self._storage_name.suffix == 'g':
                self._logger.info(
                    f'Observation {obs_id}: using first slice of '
                    f'{self._science_fqn}.'
                )
                # TODO - fix this
                if self._storage_name.file_name.endswith('.fz'):
                    temp_science_f_name = (
                        self._storage_name.file_name.replace(
                            '.fz', '_slice.fz'
                        )
                    )
                else:
                    temp_science_f_name = (
                        self._storage_name.file_name.replace(
                            '.fits', '_slice.fits'
                        )
                    )

                slice_cmd = (
                    f'fitscopy {self._science_fqn}[*][*,*,1:1,1:1] '
                    f'{temp_science_f_name}'
                )
                self._exec_cmd_chdir(temp_science_f_name, slice_cmd)
                science_fqn = f'{self._working_dir}/{temp_science_f_name}'
                delete_list.append(science_fqn)

            if num_extensions >= 4:
                self._logger.info(
                    f'Observation {obs_id}: using slice for zoom preview of '
                    f'{self._science_fqn}.'
                )
                zoom_science_f_name = self._storage_name.file_name.replace(
                    '.fits', '_zoom.fits'
                )
                slice_cmd = (
                    f'fitscopy {self._science_fqn}[4][*,*,1:1] '
                    f'{zoom_science_f_name}'
                )
                self._exec_cmd_chdir(zoom_science_f_name, slice_cmd)
                zoom_science_fqn = (
                    f'{self._working_dir}/{zoom_science_f_name}'
                )
                delete_list.append(zoom_science_fqn)

        elif self._instrument in [md.Inst.MEGACAM, md.Inst.MEGAPRIME]:
            rotate_param = '-rotate 180'
            # SF - 09-04-20 - mosaic MEFs i.e. number of HDU > 1
            mode_param = ''
            if num_extensions > 1:
                mode_param = '-mode none'

        # SF - 08-04-20 - change to minmax for 'm' files instead of zscale
        # 'm' is equivalent to 'MASK'
        if self._storage_name.suffix == 'm' or self._obs_type == 'MASK':
            scale_param = 'minmax'

        # set up the correct parameters to the ds9 command
        scope_param = 'local'
        if (
            self._instrument
            in [md.Inst.ESPADONS, md.Inst.SITELLE, md.Inst.SPIROU]
            or self._intent is ObservationIntentType.SCIENCE
        ):
            scope_param = 'global'

        # 20-03-20 - seb - always use iraf - do not trust wcs coming from
        # the data acquisition. A proper one needs processing which is often
        # not done on observations.
        mosaic_param = '-mosaicimage iraf'
        if 'scatter' in self._science_fqn:
            # SF 23-02-21
            # for scatter: just take the first HDU, the headers do not have
            # the information to display the 36 images as a mosaic. so
            # basically replacing the -mosaic iraf part and taking [1] for
            # the hdu
            mosaic_param = f'-fits {self._science_fqn}[1]'
        elif self._instrument in [
            md.Inst.SITELLE,
            md.Inst.ESPADONS,
            md.Inst.SPIROU,
        ]:
            mosaic_param = ''

        geometry = '256x521'

        count += CFHTPreview._gen_image(
            self._science_fqn,
            geometry,
            self._thumb_fqn,
            scope_param,
            rotate_param,
            mosaic_param=mosaic_param,
            scale_param=scale_param,
        )
        if count == 1:
            self._add_title(num_extensions, self._thumb_fqn, 8)
            self.add_preview(
                self._storage_name.thumb_uri,
                self._storage_name.thumb,
                ProductType.THUMBNAIL,
                ReleaseType.META,
            )
            self.add_to_delete(self._thumb_fqn)

        geometry = '1024x1024'
        if self._instrument in [md.Inst.MEGACAM, md.Inst.MEGAPRIME]:
            count += CFHTPreview._gen_image(
                self._science_fqn,
                geometry,
                self._preview_fqn,
                scope_param,
                rotate_param,
                mosaic_param=mosaic_param,
                mode_param=mode_param,
                scale_param=scale_param,
            )
        else:
            count += CFHTPreview._gen_image(
                self._science_fqn,
                geometry,
                self._preview_fqn,
                scope_param,
                rotate_param,
                mosaic_param=mosaic_param,
                scale_param=scale_param,
            )
        if count == 2:
            self._add_title(num_extensions, self._preview_fqn, offset=2)
            self.add_preview(
                self._storage_name.prev_uri,
                self._storage_name.prev,
                ProductType.PREVIEW,
                ReleaseType.DATA,
            )
            self.add_to_delete(self._preview_fqn)

        mosaic_param = '-fits'
        zoom_param = '1'
        scope_param = 'global'
        # set zoom parameters
        if 'scatter' in self._science_fqn:
            mosaic_param = f'-fits {zoom_science_fqn}[1]'
            zoom_science_fqn = ''
            pan_param = ''
        elif self._instrument in [md.Inst.ESPADONS, md.Inst.SPIROU]:
            pan_param = ''
        elif self._instrument is md.Inst.WIRCAM:
            pan_param = '-pan 484 -484 image'
            if self._storage_name.suffix == 'g':
                pan_param = ''
                zoom_param = 'to fit'
        elif self._instrument in [md.Inst.MEGACAM, md.Inst.MEGAPRIME]:
            pan_param = '-pan -9 1780'
            rotate_param = '-rotate 180'
            if num_extensions >= 23:
                rotate_param = ''
                mosaic_param = f'-fits {zoom_science_fqn}[23]'
            elif num_extensions >= 14:
                mosaic_param = f'-fits {zoom_science_fqn}[14]'
            else:
                mosaic_param = f'-fits {zoom_science_fqn}[1]'
            zoom_science_fqn = ''
        elif self._instrument is md.Inst.SITELLE:
            pan_param = '-pan -512 1544'
        count += CFHTPreview._gen_image(
            zoom_science_fqn,
            geometry,
            self._zoom_fqn,
            scope_param,
            rotate_param,
            zoom_param,
            pan_param,
            mosaic_param=mosaic_param,
            scale_param=scale_param,
        )
        if os.path.exists(self._zoom_fqn):
            CFHTPreview._gen_square(self._zoom_fqn)
            if count == 3:
                self.add_preview(
                    self._storage_name.zoom_uri,
                    self._storage_name.zoom,
                    ProductType.PREVIEW,
                    ReleaseType.DATA,
                )
                self.add_to_delete(self._zoom_fqn)
        return count

    def _do_spirou_bintable(self):
        label = f'{self._storage_name.product_id}: {self._target_name}'
        self._logger.debug(f'Generating {label} plot')

        df = Table.read(self._science_fqn)
        plt.figure(figsize=(10.24, 10.24), dpi=100)
        found = False
        # From Chris Usher at CFHT - 18-10-23 - the change from 'Velocity' to 'RV' would have happened as part of the
        # APERO (SPIRou DRS) version update, where the final products for processed data moved from something 
        # generated by CFHT to something APERO creates itself.
        for column_name in ['Velocity', 'RV']:
            try:
                plt.plot(df[column_name], df['Combined'])
                found = True
                break
            except KeyError:
                continue
        if not found:
            self._logger.debug(df.info())
            raise mc.CadcException('Unexpected column names.')

        plt.title(label, weight='bold', color='m')
        plt.xlabel('Radial Velocity (km/s)')
        plt.ylabel('Weighted mean echelle order')
        plt.savefig(self._preview_fqn, format='jpg')
        plt.close()
        return self._save_figure()

    def _do_spirou_intensity_spectrum(self):
        self._logger.debug('Begin _do_spirou_intensity_spectrum')
        self._hdu_list = fits.open(self._science_fqn)
        self._ext = 0

        # Polarization scale factor

        if self._storage_name.suffix in ['e', 't']:
            sw2d = self._hdu_list['WaveAB'].data.copy()  # wavelength array (nm)
            si2d = self._hdu_list['FluxAB'].data.copy()  # intensity array (normalized)
            del self._hdu_list['WaveAB'].data
            del self._hdu_list['FluxAB'].data
            sw = np.ravel(sw2d)
            si = np.ravel(si2d)

        if self._storage_name.suffix == 'p':
            sw2d = self._hdu_list['WaveAB'].data.copy()  # wavelength array (nm)
            si2d = self._hdu_list['StokesI'].data.copy()  # intensity array (normalized)
            sp2d = self._hdu_list['Pol'].data.copy()  # Pol Stokes array
            del self._hdu_list['WaveAB'].data
            del self._hdu_list['StokesI'].data
            del self._hdu_list['Pol'].data
            sw = np.ravel(sw2d)
            si = np.ravel(si2d)
            sp = np.ravel(sp2d)
            pScale = 5.0 * max(si)

        if self._storage_name.suffix == 's':
            # using uniform wavelength bins
            self._ext = 1
            sw = self._hdu_list[self._ext].data.copy().field(0)
            si = self._hdu_list[self._ext].data.copy().field(1)

        npix = sw.shape[0]
        swa = 10.0 * sw
        sia = np.arange(0.0, npix, 1.0)
        spa = None
        if self._storage_name.suffix == 'p':
            spa = np.arange(0.0, npix, 1.0)
        # determine upper/lower y limits for two planes from intensity values
        for i in range(sia.shape[0]):
            sia[i] = float(si[i])
            if self._storage_name.suffix == 'p':
                spa[i] = float(sp[i]) * pScale  # increase polarization scale
        fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
        result = 0
        if self._subplot(fig, swa, sia, spa, 15000.0, 15110.0, 1, 15030.0, 15030.0, 'Stokes spectrum'):
            if self._subplot(fig, swa, sia, spa, 22940.0, 23130.0, 2, 22990.0, 22990.0, 'Stokes spectrum'):
                plt.tight_layout()
                plt.savefig(self._preview_fqn, format='jpg')
                plt.close()
                result = self._save_figure()
        return result

    def _exec_cmd_chdir(self, temp_file, cmd):
        orig_dir = os.getcwd()
        try:
            os.chdir(self._working_dir)
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            mc.exec_cmd(cmd)
        finally:
            os.chdir(orig_dir)

    @staticmethod
    def _gen_image(
        in_science_fqn,
        geometry,
        save_fqn,
        scope_param,
        rotate_param,
        zoom_param='to fit',
        pan_param='',
        mosaic_param='',
        mode_param='-mode none',
        scale_param='',
    ):
        # 20-03-20 - seb - always use iraf - do not trust wcs coming from the
        # data acquisition. A proper one needs processing which is often not
        # done on observations.
        cmd = (
            f'xvfb-run -a ds9 {mosaic_param} {in_science_fqn} '
            f'{pan_param} '
            f'-geometry {geometry} '
            f'{rotate_param} '
            f'-scale squared '
            f'-scale mode {scale_param} '
            f'-scale scope {scope_param} '
            f'-scale datasec yes '
            f'-invert '
            f'{mode_param} '
            f'-view colorbar no '
            f'-zoom {zoom_param} '
            f'-saveimage jpeg {save_fqn} '
            f'-quit'
        )
        mc.exec_cmd(cmd, timeout=900)  # wait 15 minutes till killing
        count = 0
        if os.path.exists(save_fqn):
            count = 1
        return count

    def _add_title(self, num_extensions, in_fqn, font_size=16, offset=0):
        if (
            self._instrument in [md.Inst.MEGACAM, md.Inst.MEGAPRIME]
            and num_extensions < 36
        ):
            # SF 02-26-21
            # add an option to the ds9 command: -grid title text {this file
            # has only XX HDUs}`, only for MegaPrime files below a threshold
            # of 36 extensions. I couldn't get ds9 to work, so use PIL
            # instead.
            title = (
                f'{self._storage_name.file_name} has only '
                f'{num_extensions} HDUs'
            )
            image = Image.open(in_fqn)
            width, height = image.size
            draw = ImageDraw.Draw(image)
            fpath = Path(mpl.get_data_path(), 'fonts/ttf/DejaVuSans-Bold.ttf')
            font = ImageFont.truetype(fpath.as_posix(), font_size)
            ignore_left, top, ignore_right, bottom = font.getbbox(title)
            text_height = bottom - top
            text_length = draw.textlength(title)
            margin = 10
            x = width / 2 - text_length / 2 - margin * offset
            y = text_height + margin
            # text is black
            draw.text((x, y), title, (0, 0, 0), font=font)
            image.save(in_fqn)

    @staticmethod
    def _gen_square(f_name):
        im = Image.open(f_name)
        min_size = 1024
        extent = max(min_size, im.size[0], im.size[1])
        # fill color is white
        new_im = Image.new('RGB', (extent, extent), (255, 255, 255, 0))
        new_im.paste(
            im,
            (
                int((extent - im.size[0]) / 2),
                int((extent - im.size[1]) / 2),
            ),
        )
        new_im.save(f_name)

    def _gen_thumbnail(self):
        self._logger.debug(
            f'Generating thumbnail for file {self._science_fqn}.'
        )
        count = 0
        if os.path.exists(self._preview_fqn):
            thumb = image.thumbnail(
                self._preview_fqn, self._thumb_fqn, scale=0.25
            )
            if thumb is not None:
                count = 1
            thumb.clf()
        else:
            self._logger.warning(
                f'Could not find {self._preview_fqn} for thumbnail '
                f'generation.'
            )
        return count

    def _sitelle_calibrated_cube(self):
        self._logger.debug(
            f'Do sitelle calibrated cube preview augmentation with '
            f'{self._science_fqn}'
        )
        # from genSiteprevperplane.py
        self._hdu_list = fits.open(self._science_fqn)
        self._ext = 0

        # Make a RGB colour image if it's a calibrated 3D cube
        # scan through cube to look for strongest lines
        data = self._hdu_list[self._ext].data.copy()
        head = self._hdu_list[self._ext].header
        self._logger.debug(f'{data.shape}, {data.size}')

        # trim off ends to make 2048x2048
        data = data[:, 8:2056]

        # trim off 30% of spectral edges - might be noisy
        nspecaxis = data.shape[0]
        numedgechannels = int(0.15 * nspecaxis)
        self._logger.debug(f'{numedgechannels}')

        data[:numedgechannels, :, :] = 0.0
        data[(-1 * numedgechannels) :, :, :] = 0.0
        nspecaxis = data.shape[0]
        nspataxis = data.shape[1] * data.shape[2]
        self._logger.debug(
            f'{nspecaxis}, {nspataxis}, {data.size}, {data.shape}'
        )
        data2d = np.reshape(data, (nspecaxis, -1))
        self._logger.debug(f'{data2d.shape}')

        for k in range(nspecaxis):
            medianvswavenumber = np.median(data2d[k, :])
            data2d[k, :] = data2d[k, :] - medianvswavenumber
        meanbgsubvswavenumber = np.mean(data2d, axis=1)

        self._logger.debug(
            f'{meanbgsubvswavenumber}, {meanbgsubvswavenumber.shape}'
        )
        indexmax1 = np.nanargmax(meanbgsubvswavenumber)
        self._logger.debug(f'{indexmax1}, {meanbgsubvswavenumber[indexmax1]}')

        # remove 7 channels around strongest line
        indexmax1lo = indexmax1 - 3
        indexmax1hi = indexmax1 + 3
        meanbgsubvswavenumber[indexmax1lo:indexmax1hi] = 0.0
        indexmax2 = np.nanargmax(meanbgsubvswavenumber)
        self._logger.debug(f'{indexmax2}, {meanbgsubvswavenumber[indexmax2]}')

        # remove 7 channels around second strongest line
        indexmax2lo = indexmax2 - 3
        indexmax2hi = indexmax2 + 3
        meanbgsubvswavenumber[indexmax2lo:indexmax2hi] = 0.0
        indexmax1loline = indexmax1 - 1
        indexmax1hiline = indexmax1 + 1
        indexmax2loline = indexmax2 - 1
        indexmax2hiline = indexmax2 + 1
        self._logger.debug(
            f'{indexmax1loline}, {indexmax1hiline}, {indexmax2loline}, '
            f'{indexmax2hiline}'
        )
        self._logger.debug(f'{meanbgsubvswavenumber}')

        w = np.where(meanbgsubvswavenumber > 0.0)
        self._logger.debug(f'{w[0]}')

        head['NAXIS1'] = 1024
        head['NAXIS2'] = 1024

        head256 = head
        head256['NAXIS1'] = 256
        head256['NAXIS2'] = 256

        # Make two line images in 3 different sizes
        dataline1 = data[indexmax1loline:indexmax1hiline, :, :]
        data2dline1 = np.mean(dataline1, axis=0)
        self._logger.debug(f'{data2dline1.shape}')

        dataline2 = data[indexmax2loline:indexmax2hiline, :, :]
        data2dline2 = np.mean(dataline2, axis=0)
        self._logger.debug(f'{data2dline2.shape}')

        # Make "continuum" image with two strongest lines removed in 3
        # different sizes and add this to line image so whole image not green
        datanolines = data[w[0], :, :]
        data2dcont = np.mean(datanolines, axis=0)

        data2dline1pluscont = data2dline1 + data2dcont
        data2dline2pluscont = data2dline2 + data2dcont
        self._logger.debug(
            f'{np.mean(data2dline1)}, '
            f'{np.mean(data2dline1pluscont)}, '
            f'{np.mean(data2dline2pluscont)}, '
            f'{np.mean(data2dcont)}'
        )

        self._create_rgb_inputs(
            data2dline1pluscont,
            head,
            head256,
            'imageline1size1024.fits',
            'imageline1size256.fits',
            'imageline1zoom1024.fits',
        )
        self._create_rgb_inputs(
            data2dline2pluscont,
            head,
            head256,
            'imageline2size1024.fits',
            'imageline2size256.fits',
            'imageline2zoom1024.fits',
        )
        self._create_rgb_inputs(
            data2dcont,
            head,
            head256,
            'imagecontsize1024.fits',
            'imagecontsize256.fits',
            'imagecontzoom1024.fits',
        )

        os.system("pwd")
        del data
        del datanolines

        self._create_rgb(
            'imageline1size1024.fits',
            'imageline2size1024.fits',
            'imagecontsize1024.fits',
            self._preview_fqn,
        )
        self._create_rgb(
            'imageline1size256.fits',
            'imageline2size256.fits',
            'imagecontsize256.fits',
            self._thumb_fqn,
        )
        self._create_rgb(
            'imageline1zoom1024.fits',
            'imageline2zoom1024.fits',
            'imagecontzoom1024.fits',
            self._zoom_fqn,
        )
        self.add_to_delete('./imageline1size1024.fits')
        self.add_to_delete('./imageline1size256.fits')
        self.add_to_delete('./imageline1zoom1024.fits')
        self.add_to_delete('./imageline2size1024.fits')
        self.add_to_delete('./imageline2size256.fits')
        self.add_to_delete('./imageline2zoom1024.fits')
        self.add_to_delete('./imagecontsize1024.fits')
        self.add_to_delete('./imagecontsize256.fits')
        self.add_to_delete('./imagecontzoom1024.fits')
        self.add_preview(
            self._storage_name.thumb_uri,
            self._storage_name.thumb,
            ProductType.THUMBNAIL,
            ReleaseType.META,
        )
        self.add_preview(
            self._storage_name.prev_uri,
            self._storage_name.prev,
            ProductType.PREVIEW,
            ReleaseType.DATA,
        )
        self.add_to_delete(self._thumb_fqn)
        self.add_to_delete(self._preview_fqn)
        self.add_preview(
            self._storage_name.zoom_uri,
            self._storage_name.zoom,
            ProductType.PREVIEW,
            ReleaseType.DATA,
        )
        self.add_to_delete(self._zoom_fqn)
        return 3

    def _sitelle_hdf5(self):
        self._logger.debug(f'Do sitelle hdf5 preview augmentation with {self._science_fqn}')
        # Laurie Rousseau-Nepton - 11-08-22
        with h5py.File(self._science_fqn, 'r') as f:
            dataset = f.get('deep_frame')
            plt.figure(figsize=(10.24, 10.24), dpi=200)
            plt.axis('off')
            plt.imshow(np.log10(dataset), vmin=4.5, vmax=5.5)
            plt.savefig(self._preview_fqn, bbox_inches='tight', format='jpg')
        return self._save_figure()

    def _create_rgb_inputs(
        self,
        input_data,
        head,
        head256,
        preview_f_name,
        thumb_f_name,
        zoom_f_name,
    ):
        size1024 = self._rebin_factor(input_data, (1024, 1024))
        size256 = self._rebin_factor(input_data, (256, 256))
        zoom1024 = input_data[512:1536, 512:1536]
        self._logger.debug(
            f'{size1024.shape}, {size256.shape}, {zoom1024.shape}'
        )
        fits.writeto(preview_f_name, size1024, head, overwrite=True)
        fits.writeto(thumb_f_name, size256, head256, overwrite=True)
        fits.writeto(zoom_f_name, zoom1024, head, overwrite=True)

    @staticmethod
    def _create_rgb(line1_f_name, line2_f_name, cont_f_name, fqn):
        aplpy.make_rgb_image(
            [line1_f_name, line2_f_name, cont_f_name],
            fqn,
            stretch_r='linear',
            stretch_g='linear',
            stretch_b='linear',
            pmax_r=99.5,
            pmax_g=99.5,
            pmax_b=99.5,
            pmin_r=50.0,
            pmin_g=95.0,
            pmin_b=50.0,
            embed_avm_tags=False,
        )

    def _rebin_factor(self, a, new_shape):
        """
        Re-bin an array to a new shape.

        :param new_shape must be a factor of a.shape.
        """
        assert len(a.shape) == len(new_shape)
        assert not np.sometrue(np.mod(a.shape, new_shape))

        slices = [
            slice(None, None, mc.to_int(old / new))
            for old, new in zip(a.shape, new_shape)
        ]
        self._logger.debug(slices)
        return a[tuple(slices)]

    def _save_figure(self):
        self.add_to_delete(self._preview_fqn)
        count = 1
        self.add_preview(
            self._storage_name.prev_uri,
            self._storage_name.prev,
            ProductType.PREVIEW,
            ReleaseType.DATA,
        )
        count += self._gen_thumbnail()
        if count == 2:
            self.add_preview(
                self._storage_name.thumb_uri,
                self._storage_name.thumb,
                ProductType.THUMBNAIL,
                ReleaseType.META,
            )
            self.add_to_delete(self._thumb_fqn)
        return count


def visit(observation, **kwargs):
    # default value of SITELLE covers the only known case of undefined instrument, which occurs when a SITELLE HDF5
    # file has no "attrs"
    previewer = CFHTPreview(
        observation.instrument.name if observation.instrument is not None else md.Inst.SITELLE.value,
        observation.intent,
        observation.type,
        observation.target,
        **kwargs,
    )
    # SF - 23-02-21
    # No previews for 'frpts' files.
    if '.frpts.' in previewer.storage_name.file_name:
        result = observation
    else:
        result = previewer.visit(observation)
    return result
