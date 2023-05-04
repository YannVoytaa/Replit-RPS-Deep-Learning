import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(9, 3)

  def forward(self, x):
    x = self.fc1(x)
    x = F.softmax(x, dim=0)
    return x


net = Net()

criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=1)

to_choose = ["R", "P", "S"]


def winner(opp_play):
  if opp_play == "R":
    return "P"
  if opp_play == "P":
    return "S"
  return "R"


def get_tensor(play):
  if play == "R":
    return torch.tensor([1.0, 0.0, 0.0])
  if play == "P":
    return torch.tensor([0.0, 1.0, 0.0])
  return torch.tensor([0.0, 0.0, 1.0])


my_last_3rd = get_tensor("R")
my_last_2nd = get_tensor("R")
my_last = get_tensor("R")


def player(prev_play, opponent_history=[]):
  global my_last
  global my_last_2nd
  global my_last_3rd
  idx = torch.tensor([0.0, 0.0, 0.0])
  if prev_play != "":
    input = get_tensor(
      opponent_history[-1] if len(opponent_history) > 0 else None)
    idx = get_tensor(prev_play)
    optimizer.zero_grad()
    outputs = net(torch.cat((input, my_last_2nd, my_last_3rd)))

    loss = criterion(outputs, idx)

    loss.backward()
    optimizer.step()
    #if len(opponent_history) < 50:
    #  print('loss', loss)
  opponent_history.append(prev_play)
  last_pred = net(torch.cat((idx, my_last, my_last_2nd)))
  #if len(opponent_history) < 50:
  #  print(last_pred)
  opp_guess = to_choose[torch.multinomial(last_pred, 1).item()]
  my_guess = winner(opp_guess)
  my_last_3rd = my_last_2nd
  my_last_2nd = my_last
  my_last = get_tensor(my_guess)
  #if len(opponent_history) < 50:
  #  print(opp_guess, my_guess)
  return my_guess
