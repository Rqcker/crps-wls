# LUCAS 2018 topsoil data

The simulation experiments draw ground-truth SOC values from the LUCAS 2018
topsoil survey. Its licence does not allow redistribution here. Request the
dataset (free for research, usually approved within a few working days) from
the European Soil Data Centre:

- https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data

From the download, place the main CSV in this directory:

```
data/LUCAS-SOIL-2018.csv
```

The loader (`src/dataset.py`) reads the `OC` column (soil organic carbon,
g/kg, 0-20 cm depth) and keeps mineral soils (OC <= 100 g/kg), the domain on
which the sensor parameters were measured.
