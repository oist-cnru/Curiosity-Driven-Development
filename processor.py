#%%
import torch
import pybullet as p
import numpy as np
from time import sleep
from random import uniform, choice

from utils import (
    task_map, shape_map, color_map, make_objects_and_task, print,
    Goal, Obs, empty_goal, opposite_relative_to
)
from utils_submodule import pad_zeros
from arena import Arena, get_physics


class Processor:
    """
    A processor connects agents and arenas (robots and environments).
    """

    def __init__(
        self, args, arena_1, arena_2,
        tasks_and_weights=[(0, 1)],
        objects=1, colors=[0], shapes=[0],
        parenting=True, linestyle='-', full_name=''
    ):
        """
        Initialize processor with arenas and allowed goal types.
        """
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


    def begin(self, test=False, verbose=False, set_positions=None, set_goal=None):
        """
        Begin a new episode with a goal and object setup.
        """
        self.steps = 0

        if set_goal is None:
            goal_task, self.current_objects_1, self.current_objects_2 = make_objects_and_task(
                num_objects=self.objects,
                allowed_tasks_and_weights=self.tasks_and_weights,
                allowed_colors=self.colors,
                allowed_shapes=self.shapes,
                test_train_num=self.args.test_train_num,
                test=test
            )
        else:
            goal_task, self.current_objects_1, self.current_objects_2 = set_goal[0], set_goal[1], set_goal[2]

        goal_color, goal_shape = self.current_objects_1[0]

        if goal_task.name == 'FREEPLAY':
            goal_color = goal_task
            goal_shape = goal_task

        self.goal = Goal(goal_task, goal_color, goal_shape, self.parenting)

        self.arena_1.begin(self.current_objects_1, self.goal, self.parenting, set_positions=set_positions)
        if not self.parenting:
            self.arena_2.begin(self.current_objects_2, self.goal, self.parenting, set_positions=set_positions)

        self.feedback_voice_1 = empty_goal
        self.feedback_voice_2 = empty_goal

        if verbose:
            print(self)


    def __str__(self):
        """
        String representation showing current goal and object colors/shapes.
        """
        to_return = '\n\nSHAPE-COLORS (1):\t{}'.format(
            ['{} {}'.format(color, shape) for color, shape in self.current_objects_1]
        )
        if not self.parenting:
            to_return += '\nSHAPE-COLORS (2):\t{}'.format(
                ['{} {}'.format(color, shape) for color, shape in self.current_objects_2]
            )

        self.goal.make_texts()
        to_return += '\nGOAL:\t{} ({})'.format(self.goal.char_text, self.goal.human_text)
        return to_return


    def get_arena(self, agent_1=True):
        """
        Return arena corresponding to agent 1 or 2.
        """
        if agent_1:
            return self.arena_1
        if self.parenting:
            return None
        return self.arena_2


    def obs(self, agent_1=True):
        """
        Gather vision, touch, proprioception, and feedback voice.
        """
        arena = self.get_arena(agent_1)
        if arena is None:
            return Obs(torch.zeros((1, self.args.image_size, self.args.image_size, 4)), None, None, None)

        vision = arena.photo_for_agent()
        vision = torch.from_numpy(vision).float().unsqueeze(0)

        touched = [0] * self.args.touch_shape
        for _, object_dict in arena.objects_touch.items():
            for i, (_, value) in enumerate(object_dict.items()):
                touched[i] += value

        touch = torch.tensor([touched]).float()

        if self.args.tanh_touch:
            tanh_touch = torch.tanh((touch - 0.5) * 10)
            tanh_touch = (tanh_touch + 1) / 2
            touch = tanh_touch

        joint_angles = arena.get_joint_angles()
        joint_angles_regularized = [
            (1 + opposite_relative_to(
                angle,
                getattr(self.args, f'min_joint_{key}_angle'),
                getattr(self.args, f'max_joint_{key}_angle'))) / 2
            for key, angle in joint_angles.items()
        ]

        joint_speeds = arena.get_joint_speeds()
        joint_speeds_regularized = []
        for key, angle in joint_speeds.items():
            joint_speed = opposite_relative_to(
                angle, 
                -self.args.max_joint_speed,
                self.args.max_joint_speed)
            joint_speeds_regularized.append((1 + joint_speed)/2)

        prop = joint_angles_regularized + joint_speeds_regularized
        prop = torch.tensor([prop]).float()

        feedback_voice = self.feedback_voice_1 if agent_1 else self.feedback_voice_2

        return(Obs(vision, touch, prop, self.goal, feedback_voice))


    def act(self, wheels_joints, agent_1=True, verbose=False, sleep_time=None):
        """
        Apply a set of motor commands in the arena.
        """
        arena = self.get_arena(agent_1)
        if arena is None:
            return None, None

        left_wheel_speed = wheels_joints[0].item()
        right_wheel_speed = wheels_joints[1].item()
        joint_speeds = {i - 1: wheels_joints[i] for i in range(2, len(wheels_joints))}

        if verbose:
            print(f'\n\nStep {self.steps}:')
            print('Wheels: {}, {}.'.format(round(left_wheel_speed, 2), round(right_wheel_speed, 2)))
            print('Joints:', [f'{key}: {round(value.item(), 2)}' for key, value in joint_speeds.items()])

        arena.step(left_wheel_speed, right_wheel_speed, joint_speeds, verbose=verbose, sleep_time=sleep_time)
        reward, win, feedback_voice = arena.rewards(verbose=verbose)

        if agent_1:
            self.feedback_voice_1 = feedback_voice
        else:
            self.feedback_voice_2 = feedback_voice

        return reward, win


    def step(self, wheels_joints_1, wheels_joints_2=None, verbose=False, sleep_time=None):
        """
        Take one step in the simulation, return reward, done, win.
        """
        self.steps += 1
        done = False

        reward, win = self.act(wheels_joints_1, verbose=verbose, sleep_time=sleep_time)

        if not self.parenting:
            reward_2, win_2 = self.act(wheels_joints_2, agent_1=False, verbose=verbose, sleep_time=sleep_time)
            reward = max(reward, reward_2)
            win = win or win_2

        if reward > 0:
            reward *= self.args.step_cost ** (self.steps - 1)

        end = self.steps >= self.args.max_steps

        if end and not win:
            done = True
            if self.arena_1.goal.task.name != 'FREEPLAY':
                reward += self.args.step_lim_punishment
            if verbose:
                print('Episode end!', end=' ')

        if win:
            done = True
            if verbose:
                print('Correct!', end=' ')

        if verbose and done:
            print('Done.')

        return reward, done, win


    def done(self):
        """
        End episode, clear feedback voices.
        """
        self.arena_1.end()
        if not self.parenting:
            self.arena_2.end()

        self.feedback_voice_1 = empty_goal
        self.feedback_voice_2 = empty_goal


# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from utils import args

    arena_1 = Arena(GUI=True, args=args)
    arena_2 = None

    processor = Processor(
        args, arena_1, arena_2,
        tasks_and_weights=[(0, 1)],
        objects=2,
        colors=[0, 1, 2, 3, 4, 5],
        shapes=[0],
        parenting=True
    )

    def get_images():
        rgba = processor.arena_1.photo_from_above()
        obs = processor.obs()
        rgb = obs.vision[0, :, :, :3]
        d = obs.vision[0, :, :, -1]
        return rgba, rgb, d

    def example_images(images):
        rgba, rgb, d = images

        plt.imshow(rgba)
        plt.axis('off')
        plt.show()
        plt.close()

        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
        plt.close()

        plt.imshow(d, cmap='gray', interpolation='none')
        plt.axis('off')
        plt.show()
        plt.close()

    i = 0
    while True:
        i += 1
        print('episode', i)
        processor.begin(verbose=True)
        done = False
        j = 0
        while not done:
            j += 1
            print('step', j)
            example_images(get_images())
            recommendation = torch.zeros((4,))
            print('Got recommendation:', recommendation)
            reward, done, win = processor.step(recommendation, verbose=True)
            print('Done:', done)
            obs = processor.obs()
            rgb = obs.vision[0, :, :, :3]
            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
            plt.close()
            sleep(0.1)
        print('Win:', win)
        example_images(get_images())
        processor.done()
