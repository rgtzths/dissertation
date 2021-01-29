# thesis

### Python Requirements

- Install NILMTK (requires miniconda) - https://github.com/nilmtk/nilmtk/blob/master/docs/manual/user_guide/install_user.md
- Install neuralnilm
  - First download the git - https://github.com/JackKelly/neuralnilm
  - using the environment in conda do
  - pip3 install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
  - pip3 install Lasagne==0.1
  - pip install pymongo
  - pip install h5py
  - inside the git folder now do
  - edit setup.py var "name" to "neuralnilm"
  - pip3 install .
  - in the requirements.txt remove the last 4 lines
  - run pip3 install -r requirements.txt
- Download UKDale H5 - https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip