from functools import partial
from queue import Queue

import numpy as np
import cv2

from hamiltonian import DiscreteSpace, SingleParticle
from time_evolve import VisscherPropagator as VissProp
from video_utils import VideoWriterStream, render_frames
from potentials import multiple_hard_disks
from states import coherent_state_2d


def main():
    # ------------------------
    # Simulation Setup
    # ------------------------

    # potential
    num_cols = 5
    dot_radius = 0.20
    step = 3 * dot_radius
    ys = np.arange(-7, 7, step)
    centers = [(3 + i * step, y + i % 2 * step / 2) for i in range(num_cols) for y in ys]
    rs = [dot_radius] * (len(centers))
    potential = partial(multiple_hard_disks, rs=rs, centers=centers)

    # initial state
    lam = dot_radius
    p = (1.6/lam, 0)
    xy0 = (-4, 0)
    w = (1, 1)
    init_state = partial(coherent_state_2d, p=p, xy0=xy0, w=w)

    # system and solver
    dim = 2  # spacial dimension
    support = (-6, 6)  # support region of mask_func
    grid = 100 # number of grid points along one dimension. Assumed square.
    dtype = np.float32  # datatype used for internal processing
    dt = 0.001
    sys_duration = 2

    # video arguments
    name = 'visscher_prop'
    vid_duration = 6
    fps = 30
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
    mask_grid = ham.potential(*space_vid.grid_points)

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
