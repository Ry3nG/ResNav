from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

from me5418_nav.envs import UnicycleNavEnv, EnvConfig
from me5418_nav.viz.plotting import draw_env


class KeyController:
    def __init__(self, v_step=0.1, w_step=0.2):
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.v_step = v_step
        self.w_step = w_step

    def on_key(self, event):
        if event.key == "up":
            self.v_cmd += self.v_step
        elif event.key == "down":
            self.v_cmd -= self.v_step
        elif event.key == "left":
            self.w_cmd += self.w_step
        elif event.key == "right":
            self.w_cmd -= self.w_step
        elif event.key == " ":  # space brake
            self.v_cmd = 0.0
            self.w_cmd = 0.0
        elif event.key == "r":
            self.v_cmd = 0.0
            self.w_cmd = 0.0
            raise ResetException()
        elif event.key == "q":
            raise QuitException()


class ResetException(Exception):
    pass


class QuitException(Exception):
    pass


def draw(env: ManualNavEnv, ax):
    draw_env(env, ax)
    ax.set_title("Manual Control: arrows=velocity, space=brake, r=reset, q=quit")


def main():
    cfg = EnvConfig()
    env = UnicycleNavEnv(cfg)
    controller = KeyController()

    fig, ax = plt.subplots(figsize=(6, 6))
    cid = fig.canvas.mpl_connect("key_press_event", controller.on_key)

    while True:
        try:
            obs, _ = env.reset()
            done = False
            draw(env, ax)
            plt.pause(0.001)
            while not done:
                action = np.array([controller.v_cmd, controller.w_cmd], dtype=float)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                draw(env, ax)
                plt.pause(env.cfg.dt)
        except ResetException:
            env.reset()
            draw(env, ax)
            plt.pause(0.001)
            continue
        except QuitException:
            break

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)


if __name__ == "__main__":
    main()
