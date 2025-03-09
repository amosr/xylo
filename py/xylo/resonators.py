# approximate quarter-wave stopped resonators
import math

import typing

class Config(typing.NamedTuple):
  temp: float = 20
  correction: float = 0.61
  radius: float = 0.018

def speed_of_sound(temp):
  return 331.3 * math.sqrt(1 + temp / 273.15)
  # return 331.3 * math.sqrt((1 + (temp + 273.15)) / 273.15)

# 346.129
# speed_of_sound(25)

def freq_of_length(length, config: Config):
  return speed_of_sound(config.temp) / (4 * (length + config.correction * config.radius))

def length_of_freq(freq, config: Config):
  return (speed_of_sound(config.temp) * (1/freq) * 1/4) - config.correction * config.radius

