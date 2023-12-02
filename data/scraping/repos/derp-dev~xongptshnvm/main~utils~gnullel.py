#from langchain import def get_absolute_url(self):
  # from django.core.urlresolvers import reverse
  # return reverse('', kwargs={'pk': self.pk})
import sys, time
from mpi4pi import MPI

comm = MPI.COMM_WORLD # communicator
rank = comm.Get_rank()
size = comm.Get_size() # number of processes
root = 0 # prints from root at end of program

steps = int(sys.argv[1], 10) # number of steps to take

def gnullel(steps):
  step = 1.0 / steps
  sum = 0.0
  for i in range(rank, steps, size):
    x = (i + 0.5) * step
    sum += 4.0 / (1.0 + x * x)
    return sum * step

start = time.time()
i = comm.reduce(gnullel(steps), op=MPI.SUM, root=root)
end = time.time()