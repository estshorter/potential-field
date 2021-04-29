import numpy as np

from animation import Animation_robot


class Obstacle:
    def __init__(self, x, y, size) -> None:
        self.x = x
        self.y = y
        self.size = size


class TwoWheeledRobot:
    def __init__(self, init_x, init_y, init_th) -> None:
        self.x = init_x
        self.y = init_y
        self.th = init_th
        self.u_x = 0.0
        self.u_y = 0.0

        self.traj_x = [init_x]
        self.traj_y = [init_y]

    # xi: [x, y, theta]
    # u:[u_th, u_v]
    @staticmethod
    def state_equation(xi, u):
        dxi = np.empty(3)
        dxi[0] = u[1] * np.cos(xi[2])
        dxi[1] = u[1] * np.sin(xi[2])
        dxi[2] = u[0]
        return dxi

    def update_state(self, u_x, u_y, dt):
        self.u_x = u_x
        self.u_y = u_y

        next_x = u_x * dt + self.x
        next_y = u_y * dt + self.y

        self.traj_x.append(next_x)
        self.traj_y.append(next_y)

        self.x = next_x
        self.y = next_y

        return self.x, self.y, self.th


class ConstGoal:
    def __init__(self) -> None:
        self.traj_g_x = []
        self.traj_g_y = []

    def calc_goal(self, time_step):
        g_x = g_y = 10.0
        # if time_step <= 100:
        #     g_x = 10.0
        #     g_y = 10.0
        # else:
        #     g_x = -10.0
        #     g_y = -10.0

        self.traj_g_x.append(g_x)
        self.traj_g_y.append(g_y)

        return g_x, g_y


class PotentialField:
    def __init__(self) -> None:
        self.gain_attr = 0.2
        self.gain_rep = 0.2
        pass

    def calc_input(self, g_x, g_y, state, obstacles):
        term_attr = self._calc_attractive_term(g_x, g_y, state)
        term_rep = self._calc_repulsive_term(state, obstacles)
        input = self.gain_attr * term_attr + self.gain_rep * term_rep
        return input[0], input[1]

    def _calc_repulsive_term(self, state, obstacles):
        term = np.zeros(2)
        for obs in obstacles:
            dist_sqrd = (state.x - obs.x) ** 2 + (state.y - obs.y) ** 2
            denom = dist_sqrd ** 1.5
            term[0] += (state.x - obs.x) / denom
            term[1] += (state.y - obs.y) / denom
        return term

    def _calc_attractive_term(self, g_x, g_y, state):
        term = np.zeros(2)
        term[0] = g_x - state.x
        term[1] = g_y - state.y
        return term


class MainController:
    def __init__(self) -> None:
        self.dt = 0.1

        self.robot = TwoWheeledRobot(0.0, 0.0, 0)
        self.goal_maker = ConstGoal()
        self.planner = PotentialField()

        # self.obstacles = [
        #     Obstacle(4, 1, 0.25),
        #     Obstacle(0, 4.5, 0.25),
        #     Obstacle(3, 4.5, 0.25),
        #     Obstacle(5, 3.5, 0.25),
        #     Obstacle(7.5, 9.0, 0.25),
        # ]
        self.obstacles = []
        for _ in range(20):
            x = np.random.randint(0, 10)
            y = np.random.randint(0, 10)
            size = 0.25
            self.obstacles.append(Obstacle(x, y, size))

    def run(self):
        time_step = 0
        goal_th = 0.5
        goal_th_sqrd = goal_th ** 2
        max_timestep = 500

        while True:
            g_x, g_y = self.goal_maker.calc_goal(time_step)

            u_x, u_y = self.planner.calc_input(g_x, g_y, self.robot, self.obstacles)

            self.robot.update_state(u_x, u_y, self.dt)

            dist_to_goal = (g_x - self.robot.x) ** 2 + (g_y - self.robot.y) ** 2

            if dist_to_goal < goal_th_sqrd:
                break
            time_step += 1
            if time_step >= max_timestep:
                break

        return (
            self.robot.traj_x,
            self.robot.traj_y,
            self.goal_maker.traj_g_x,
            self.goal_maker.traj_g_y,
            self.obstacles,
        )


def main():
    animation = Animation_robot()
    animation.fig_set()

    controller = MainController()
    (
        traj_x,
        traj_y,
        traj_g_x,
        traj_g_y,
        obstacles,
    ) = controller.run()

    animation.func_anim_plot(traj_x, traj_y, traj_g_x, traj_g_y, obstacles)


if __name__ == "__main__":
    main()
