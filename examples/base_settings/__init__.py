from .pendulum import get_settings as pendulum_settings
from .inv_pendulum_swing_up import get_settings as inv_pendulum_su_settings
from .inv_double_pendulum import get_settings as inv_double_pendulum_settings
from .continuous_mountain_car import get_settings as continuous_mountain_car_settings


settings_dict = {"InvPendulumSwingUp": inv_pendulum_su_settings,
                 "InvDoublePendulum": inv_double_pendulum_settings,
                 "ContinuousMountainCar": continuous_mountain_car_settings,
                 "Pendulum": pendulum_settings}

__all__ = ["pendulum_settings", "inv_pendulum_su_settings",
           "inv_double_pendulum_settings", "continuous_mountain_car_settings",
           "settings_dict"]
