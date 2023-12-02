from pymatgen.core.structure import Structure
from ase.io import read, write
from ase import Atoms
from ase.visualize import view
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
import re
geom_file_ZnO = read("ZnO_opt.cif",format='cif')
geom_file_Pd = read("Pd_opt.cif", format='cif')
#view(geom_file)

#write("ZnO_bulk.cif", geom_file)

testview=(geom_file_ZnO)
view(testview)
ZnO = Structure.from_file("ZnO_opt.cif")
Pd = Structure.from_file("Pd_opt.cif")
sub_analyzer = SubstrateAnalyzer()
sub_analyzer.calculate(film=ZnO, substrate=Pd)

matches = list(sub_analyzer.calculate(film=ZnO,substrate=Pd))
for match in matches:
    print(match.match_area, match.von_mises_strain)

print(matches[0].as_dict())

zsl = ZSLGenerator(max_area=400, max_length_tol=0.005)

cib = CoherentInterfaceBuilder (film_structure=ZnO,
                                substrate_structure=Pd,
                                film_miller=(0,0,1),
                                substrate_miller=(1,1,1),
                                zslgen=zsl)

print(cib.terminations)
cib_list=[]
for termination in cib.terminations:
    cib_list.append(termination)
    print(match.match_area, match.von_mises_strain)
print(cib_list)

for cib_val in cib_list:
    interfaces=list(cib.get_interfaces(termination= (cib_val), film_thickness=2, substrate_thickness=2))
    for iinterface in interfaces:
      interface=iinterface
      #nterface.translate_sites(range(len(interface)),[0,0,0])
      print("Done!")
      print(interface)
      #Seems to be fine here: going up in 0.05 in fractional coordinates
   
      positions = [site.coords for site in interface.sites]
      symbols = [site.species_string for site in interface.sites]
   
   
      lattice = interface.lattice.matrix
   
      interface_atoms = Atoms(symbols=symbols, positions=positions, cell=lattice)
   
      interface_atoms.translate([0, 0, 0])
   
      print("Our interface has been created!")
      print(interface_atoms)
      cib_val_str = '_'.join(str(val) for val in cib_val)
      # looking for directories! Oh no! Can be fixed using 're'.
      cib_val_str = re.sub(r'[/:]', '-', cib_val_str)
   
      cif_fileout = 'interfaceZn%s.cif' % cib_val_str
      aims_fileout = 'interfaceZn%s.in' % cib_val_str
      print("Writing cif file ", cif_fileout)
      write(cif_fileout, interface_atoms, format='cif')
      write(aims_fileout, interface_atoms, format='aims')
   
      view(interface_atoms)
      
   
   
   
   
   