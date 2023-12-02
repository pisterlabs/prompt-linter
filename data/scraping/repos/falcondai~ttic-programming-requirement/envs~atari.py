from core import GymEnv
from wrappers import GrayscaleWrapper, ScaleWrapper, MotionBlurWrapper, StackFrameWrapper

def get_atari_env(env_id):
    '''use atari game envs from OpenAI gym. `env_id` should have format `type.game_title`.'''
    parts = env_id.split('.')
    if parts[0] == 'skip':
        # atari games skipping 4 frames (3 for SpaceInvaders)
        env = GymEnv('%sDeterministic-v3' % parts[-1])
        if parts[1] == 'gray':
            return MotionBlurWrapper(GrayscaleWrapper(env), mix_coeff=0.6)
        if parts[1] == 'half':
            return MotionBlurWrapper(ScaleWrapper(GrayscaleWrapper(env), scale=0.5), mix_coeff=0.6)
        if parts[1] == 'quarter':
            return MotionBlurWrapper(ScaleWrapper(GrayscaleWrapper(env), scale=0.25), mix_coeff=0.6)
        return env
    elif parts[0] == 'stack':
        # atari games with 4 most recent frames stacked
        env = GymEnv('%sDeterministic-v3' % parts[-1])
        if parts[1] == 'gray':
            return StackFrameWrapper(GrayscaleWrapper(env), stack_frames=4)
        if parts[1] == 'half':
            return StackFrameWrapper(ScaleWrapper(GrayscaleWrapper(env), scale=0.5), stack_frames=4)
        if parts[1] == 'quarter':
            return StackFrameWrapper(ScaleWrapper(GrayscaleWrapper(env), scale=0.25), stack_frames=4)
        return env
    # atari games with no frame skipping
    return GymEnv('%sNoFrameskip-v3' % parts[0])


if __name__ == '__main__':
    from core import test_env
    import sys

    env_id = sys.argv[1]
    test_env(get_atari_env(env_id))
