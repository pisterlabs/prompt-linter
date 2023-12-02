from functools import partial
from queue import Queue

import numpy as np
import cv2

from hamiltonian import DiscreteSpace, SingleParticle, Solver
from time_evolve import VisscherPropagator as VissProp
from video_utils import VideoWriterStream, render_frames
from potentials import hard_disk
from states import coherent_state_2d


def main():
    # ------------------------
    # Simulation Setup
    # ------------------------

    # potential
    r = 1
    c = (0, 0)
    potential = partial(hard_disk, r=r, center=c)

    # initial state
    p = (10, 0)
    xy0 = (-10, 0)
    w = (0.5, 0.5)
    init_state = partial(coherent_state_2d, p=p, xy0=xy0, w=w)

    # system and solver
    dim = 2  # spacial dimension
    support = (-24, 24)  # support region of mask_func
    grid = 200  # number of grid points along one dimension. Assumed square.
    dtype = np.float64  # datatype used for internal processing
    sys_duration=8
    dt = 0.0005

    # video arguments
    name = 'scattering_circular'
    fps = 30
    vid_duration = 10
    sample_interval = int(sys_duration / vid_duration / dt / fps)
    grid_video = 720
    video_size = (grid_video, grid_video)
    fourcc_str = 'mp4v'
    extension = 'mp4'
    video_file = f"../assets/{name}.{extension}"

    # ------------------------
    # Simulation objects
    # ------------------------

    space = DiscreteSpace(dim, support, grid, dtype)
    space_vid = DiscreteSpace(dim, support, grid_video, dtype)
    ham = SingleParticle(space, potential)
    prop = VissProp(ham, init_state, dt, sample_interval)

    psit_gen = prop.evolve()

    # mask out the potential region
    mask_grid = potential(*space_vid.grid_points)

    # ------------------------
    # Run simulation and create outputs
    # ------------------------

    # initiate the frame writing pipeline
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(filename=video_file,
                          fourcc=fourcc,
                          fps=fps,
                          frameSize=video_size,
                          isColor=True)
    write_queue = Queue()
    vws = VideoWriterStream(out, write_queue)
    thread = vws.start()

    # render the frames to the write_queue
    render_frames(write_queue, psit_gen, vid_duration, fps, space_vid, mask_grid=mask_grid)

    # shutdown the thread
    vws.stop()
    thread.join()


if __name__ == '__main__':
    main()
