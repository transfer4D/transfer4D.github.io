///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#pragma once

#include <vector>
#include <DemBones/DemBonesExt.h>
#include <DemBones/DemBones.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include<Eigen/Dense>


using namespace std;
using namespace Dem;

/** Read FBX files and set: model.u, model.fv, model.w (if exist), model.m (if exist), model.nB,
	model.boneName (if exist), model.parent (if exist), model.bind (if exist), model.preMulInv (if exist), model.rotOrder (if exist), model.orient (if exist)
	@return true if success
*/
bool readNumpy(Eigen::MatrixXd vert_data,vector< vector<int> > face_data, DemBonesExt<double, float>& model);
