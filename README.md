# FDTD Electromagnetic Wave Guide Simulator

In this repository you will find some electromagnetic fields simulations using [meep](https://meep.readthedocs.io/en/master/) library.

## Usage
Install miniconda
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

```
bash miniconda.sh -b -p $HOME/miniconda
```

```
export PATH=$HOME/miniconda/bin:$PATH
```

```
source ~/.bashrc
```

```
conda create -n mp -c conda-forge pymeep
```
If it does not work, try:

```
source ~/miniconda/etc/profile.d/conda.sh
```

```
conda activate mp
```

```
python3 -c 'import meep'
```

Now you're ready to run all programs here
```
python3 <file_name.py>
```

and let the magic happen!

## Some Results

![](/assets/bend_wave_guide_photonics_Ez.gif)
![](/assets/microwave_oven.gif)
![](/assets/Ex_mode_1.png)
![](/assets/Ey_mode_1.png)
![](/assets/Ez_mode_1.png)
