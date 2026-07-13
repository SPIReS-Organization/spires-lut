# spires-lut

Reflectance lookup tables (LUTs) for the
[SPIReS](https://github.com/SPIReS-Organization) package family: create, read, and
write the Mie-scattering reflectance LUTs the inversion interpolates.

LUTs conform to the LUT boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract) —
dims `(band, solar_angle, dust_concentration, grain_size)`.

> **Status:** scaffolding only. The package layout and dependency on
> `spires-contract` are in place; the implementation is to be filled in.




this has to be at base
https://github.com/mjwolff/pyDISORT.git

and then this dir "pyrt" dir sits above it. 

To install pyRT_DISORT, simply clone the repo (using git clone https://github.com/kconnour/pyRT_DISORT.git from Terminal, or clone using your favorite GUI) and move into the directory where it was cloned. You can then install it with pip install ..You should now be able to import the package with import pyrt.

can also pip instal from biosnicar-py  https://github.com/jmcook1186/biosnicar-py/tree/master 
which houses all of the data too that i neeed. 


And then  with all of these things, 


maybe before DISORT, we can just 

git clone https://github.com/jmcook1186/biosnicar-py.git
cd biosnicar-py
pip install -r requirements.txt
pip install -e .


