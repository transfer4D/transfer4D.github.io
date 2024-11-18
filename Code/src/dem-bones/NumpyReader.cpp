#include "NumpyReader.h"
#include "LogMsg.h"
#include <Eigen/Dense>

#include <map>
#include <DemBones/MatBlocks.h>

using namespace std;
using namespace Eigen;

#define err(msgStr) {msg(1, msgStr); return false;}

bool readNumpy(MatrixXd vert_data,vector< vector<int> > face_data, DemBonesExt<double, float>& model){
	
	// Using vertices define 3D parameters
	model.nS = 1;
	model.nF = vert_data.rows()/3;
	model.nV = vert_data.cols();
	model.fStart.resize(model.nS+1);
	model.fStart(0)=0;
	model.fStart(1) = model.nF;
	model.subjectID.resize(model.nF);

	for (int s=0; s<model.nS; s++)
		for (int k=model.fStart(s); k<model.fStart(s+1); k++) model.subjectID(k)=s;

	// Set model parameters
	model.v.resize(3*model.nF,model.nV);
	model.fTime.resize(model.nF);
	for(auto i=0;i<model.fTime.size();i++){
		model.fTime[i] = 1/33;
	}

	model.v = vert_data.cast<float>();


	MatrixXd wd(0, 0);

	// Using first frame as u
	msg(1, "Adding First timestep as rest pose.\n");
	model.u.resize(model.nS*3, model.nV);
	for(int i= 0;i<3;i++)
		for(int k=0;k<model.nV;k++)
			model.u(i,k) = vert_data(i,k);
	msg(1, " Done!\n");

	// Assign faces
	model.fv = face_data;

	model.parent.resize(model.nB);
	model.bind.resize(model.nS*4, model.nB*4);
	model.preMulInv.resize(model.nS*4, model.nB*4);
	model.rotOrder.resize(model.nS*3, model.nB);
	model.orient.resize(model.nS*3, model.nB);
	model.lockM.resize(model.nB);

	// Manually set fTime to 1/25 sec
	for(auto i=0;i<model.fTime.size();i++)
		model.fTime[i] = double(i);

	return 1;
}