import typing

class Command(typing.NamedTuple):
  cmd: str
  arg: dict[str,any]

  def __str__(self):
    return self.cmd + ' ' + str.join(' ', [k + str(self.arg[k]) for k in self.arg])

class Comment(typing.NamedTuple):
  comment: str

  def __str__(self):
    return '; ' + self.comment

def c(cmd, **kwargs):
  return [Command(cmd, kwargs)]

def move(**kwargs): # X = None, Y = None, Z = None
  return c('G0', **kwargs)

def cut(**kwargs): # X = None, Y = None, Z = None, F = None, S = None):
  return c('G1', **kwargs)

def prelude():
  # Absolute position
  # Millimeters
  return c('G21') + c('G90')

class Builder:
  lines: list

  def __init__(self, pre = prelude()):
    self.lines = []
    self.lines.extend(pre)

  def extend(self, line):
    self.lines.extend(line)

  def c(self, cmd, **kwargs):
    self.extend(c(cmd, **kwargs))

  def comment(self, c):
    self.extend([Comment(c)])

  def move(self, **kwargs):
    self.extend(move(**kwargs))

  def cut(self, **kwargs):
    self.extend(move(**kwargs))

  def show(self):
    return str.join('\n', [str(l) for l in self.lines])

  def write(self, fp):
    with open(fp, 'w') as file:
      # file.writelines([str(l) for l in self.lines])
      file.write(self.show())