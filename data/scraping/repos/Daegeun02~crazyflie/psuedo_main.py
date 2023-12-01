from psuedo_commander import PsuedoCommander
from psuedo_crazyflie import crazy
from psuedo_optimizer import guidance



cf = crazy()

psueddo_commander = PsuedoCommander(cf)

psueddo_commander.start()

guidance(cf, psueddo_commander)

psueddo_commander.join()