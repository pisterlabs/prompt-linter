from .types import Config, PythonPackage


def apply_heuristics(config: Config) -> Config:
    """
    Modifies config in-place based on specific needs of
    individual python packages. May add system packages,
    move specific installs to become `pip installs` in the
    `run` section instead of `python_packages`, etc.
    """
    python_package_transforms = [
        ("cog", transform_cog),
        ("librosa", transform_librosa),
        ("clip", transform_clip),
        ("pillow", transform_pillow),
        ("midisynth", transform_midisynth),
        ("midi2audio", transform_midi2audio),
        ("music21", transform_music21),
        ("pydiffvg", transform_pydiffvg),
        ("detectron2", transform_detectron2),
        ("pesq", transform_pesq),
    ]
    for package_name, transform in python_package_transforms:
        for package in config.build.python_packages:
            if package.name.lower() == package_name:
                transform(config, package)
                break

    return config


def transform_cog(config: Config, package: PythonPackage):
    """
    cog is automatically installed inside the image.
    """
    config.build.python_packages.remove(package)


def transform_librosa(config: Config, package: PythonPackage):
    """
    librosa depends on ffmpeg and libsndfile-dev for audio
    file i/o.
    """
    config.build.system_packages |= set(["ffmpeg", "libsndfile-dev"])


def transform_clip(config: Config, package: PythonPackage):
    """
    clip has to be installed directly from openai's GitHub repo.
    """
    package.name = "git+https://github.com/openai/CLIP.git"
    package.version = None


def transform_pillow(config: Config, package: PythonPackage):
    """
    Pillow depends on some system packages for image file i/o.
    """
    config.build.system_packages |= set(["libgl1-mesa-glx", "libglib2.0-0"])


def transform_midisynth(config: Config, package: PythonPackage):
    """
    midisynth needs fluidsynth to do the actual synthesis.
    """
    config.build.system_packages |= set(["fluidsynth"])


def transform_midi2audio(config: Config, package: PythonPackage):
    """
    midi2audio needs fluidsynth and ffmpeg to synthesize midi.
    """
    config.build.system_packages |= set(["fluidsynth", "ffmpeg"])


def transform_music21(config: Config, package: PythonPackage):
    """
    music21 needs the musescore package to read musescore files,
    and fluidsynth to synthesize music scores.
    """
    config.build.system_packages |= set(["musescore", "fluidsynth"])


def transform_pydiffvg(config: Config, package: PythonPackage):
    """
    pydiffvg has to be installed from source with some very
    specific compilation flags.
    """
    config.build.python_packages.remove(package)
    config.build.run += [
        """git clone https://github.com/BachiLi/diffvg && \
cd diffvg && \
git submodule update --init --recursive && \
CMAKE_PREFIX_PATH=$(pyenv prefix) DIFFVG_CUDA=1 python setup.py install"""
    ]


def transform_detectron2(config: Config, package: PythonPackage):
    """
    detectron2 has to be installed from Facebook AI's site.
    """
    config.build.python_packages.remove(package)
    # TODO: does this only work with CUDA 10.1? do we need to
    # enforce that?
    config.build.run += [
        "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html"
    ]


def transform_pesq(config: Config, package: PythonPackage):
    """
    pesq requires numpy at install time so needs to be installed
    separately.
    """
    config.build.python_packages.remove(package)
    config.build.run += [
        f"pip install pesq=={package.version}",
    ]
