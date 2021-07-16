#setup environment
module restore
source /cm/shared/opt/intel_oneapi/2021.1.1/setvars.sh
module load nalla_pcie compiler/GCC lang/Python

# build
make fdtd_emu

# run
./fdtd_emu

# plot
mkdir -p output
mv frame* output
./plot_frames.py output 162 168
