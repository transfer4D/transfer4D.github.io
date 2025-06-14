# Transfer4D: A framework for frugal motion capture and deformation transfer
## Submission ID: 5738 

### This folder contains supplementary material for the submission:
1. 5738_Video_Summary_of_the_paper.mp4 (video explaining our approach)
2. 5738.pdf supplementary pdf containing additional details. 
3. Code: Codebase for transfer4D 
4. Code/evaluationScores contains csv files of the quantitive analysis performed in the paper. 

### Implementation Requirements 
1. embedded_deformation_graph.py requires csrc modules compiled from [[Neural Non-Rigid Tracking](https://proceedings.neurips.cc/paper/2020/hash/d93ed5b6db83be78efb0d05ae420158e-Abstract.html)], NeurIPS 2021
2. Meshlab server is required to preprocess the target complete mesh. 
3. NonRigidICP uses implementation provided in [[Lepard: Learning partial point cloud matching in rigid and deformable scenes](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Lepard_Learning_Partial_Point_Cloud_Matching_in_Rigid_and_Deformable_CVPR_2022_paper.pdf)] (CVPR 2022).
4. We use the official implementation of SSDR(Siggraph 2012) github repo name: [dem-bones](https://github.com/electronicarts/dem-bones)
5. We use the online implementation of [[Automatic Rigging and Animation of 3D Characters](https://www.cs.toronto.edu/~jacobson/seminar/baran-and-popovic-2007.pdf)] (SIGGRAPH 2007).  
6. For additional ablations NRR.py requires modules from [Lepard](https://arxiv.org/abs/2111.12591) and [[OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction](https://arxiv.org/abs/2203.07977)]

Note:- For each external requirement, if the folder is already present, replace the files in the original repo with files present in the given folder. 

## Project Setup
### Datadir Creation 
To run on the custom depth videos: Create a folder for the source object in the following format 
(a) depth: folder containing depth images as 00%d.png; (b) intrinsics.txt: 3x3 camera intrinsic matrix 


### Install Lepard: 
```
cd src/lepard
conda env create -f environment.yml
conda activate lepard
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
# Replace codes in lepard with corresponding codes of lepard_transfer4d
# Download pretrained model and place in pretrained/ from Lepard: https://github.com/rabbityl/lepard
cd ..
```

### Install dependencies
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install --extra-index-url  https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.0.1cu118
pip install scikit-image plyfile open3d numba pykdtree pynput polyscope trimesh pygel3d==0.5.2  jaxtyping pymeshlab
```


### Install PYSSDR (Dem-Bones)
```
# 1. setup directory
git clone https://github.com/shubhMaheshwari/pyssdr.git --recursive
cd pyssdr

# 2. download & place the FBXSDX under ExtLibs/

# 3. add following line at the end of CMakeLists.txt:
# install(TARGETS pyssdr LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python3.8/site-packages)

# 4. build
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(python3 -m site --user-base)
make install
cd ../..
```

### Install NonRigid-ICP
```
git clone https://github.com/rabbityl/Nonrigid-ICP-Pytorch
cd NonRigidICP

# install lietorch
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
# optional: specify GPU architectures
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"
pip install --no-build-isolation .
cd ..

# install pybind11
pip install pybind11

# install nonrigidICP
cd cxx
export MAX_JOBS=1; python setup.py install
cd ../..
```

### Install NeuralTracking
```
git clone https://github.com/DeformableFriends/NeuralTracking
cd NeuralTracking/csrc
export MAX_JOBS=1; python setup.py install
cd ../..
```

### Install Pinocchio
The code of Pinocchio is from [online implementation of Pinnochio](https://github.com/pmolodo/Pinocchio)
I've updated some of the header codes to be compatible with G++ version 9.5.0, since older version of G++ compiler is not available for Ubuntu 22.04 LTS.
For those who has lower version of compiler and wants original implementation of Pinocchio, you can pull from the original repo and use it instead.
```
cd Pinocchio
make
```


## Run

Our method requires 4 steps:  
1. Non Rigid Registration 
```
python NRR.py --datadir <path-to-data> --exp <exp-name> --ablation <ablation-name> 
```
2. Curve Skeleton Extraction 
```
python run_pygel.py --datadir <path-to-data> --exp <exp-name> --ablation <ablation-name> 
```
3. Motion Skeleton Extraction
```
python detailed_clustering_ssdr.py --datadir <path-to-data> --exp <exp-name> --ablation <ablation-name>
```
4. Skeleton Embedding
- If you don't have already aligned mesh with camera, you may use option `--align_mesh_orientation`
```
python pinnochio_runner.py --datadir <path-to-data>  --mesh <path-to-target-mesh> --exp <exp-name> --ablation <ablation-name> --align_to_depth_map --select_largest_component
```