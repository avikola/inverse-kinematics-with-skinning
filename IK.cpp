#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Spring 2019
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints are subject to IK:
// IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles
// Output: handlePositions
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  // Students should implement this.
  // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
  // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
  // Then, implement the same algorithm into this function. To do so,
  // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
  // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
  // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
  // so that code is only written once. We considered this; but it is actually not easily doable.
  // If you find a good approach, feel free to document it in the README file, for extra credit.

	real euler[3], orient[3];

	// transformations:
	vector<Mat3<real>> localTransforms(fk.getNumJoints()), globalTransforms(fk.getNumJoints());

	// translations:
	vector<Vec3<real>> localTranslate(fk.getNumJoints()), globalTranslate(fk.getNumJoints());

	// Similar to that in FK.cpp:

	for (int i = 0; i < fk.getNumJoints(); i++)
	{
		// Rotations:

		euler[0] = eulerAngles[i * 3];
		euler[1] = eulerAngles[i * 3 + 1];
		euler[2] = eulerAngles[i * 3 + 2];

		Mat3<real> R_M = Euler2Rotation(euler, fk.getJointRotateOrder(i));;

		orient[0] = fk.getJointOrient(i).data()[0];
		orient[1] = fk.getJointOrient(i).data()[1];
		orient[2] = fk.getJointOrient(i).data()[2];

		Mat3<real> R_O = Euler2Rotation(orient, RotateOrder::XYZ);

		localTransforms[i] = R_O * R_M;

		// Translations:

		localTranslate[i][0] = fk.getJointRestTranslation(i)[0];
		localTranslate[i][1] = fk.getJointRestTranslation(i)[1];
		localTranslate[i][2] = fk.getJointRestTranslation(i)[2];

		int order_j = fk.getJointUpdateOrder(i);

		globalTranslate[order_j][0] = fk.getJointRestTranslation(order_j)[0];
		globalTranslate[order_j][1] = fk.getJointRestTranslation(order_j)[1];
		globalTranslate[order_j][2] = fk.getJointRestTranslation(order_j)[2];

		if (fk.getJointParent(order_j) == -1)		// if root.
			globalTransforms[order_j] = localTransforms[order_j];
		else		
			multiplyAffineTransform4ds(
				globalTransforms[fk.getJointParent(order_j)],
				globalTranslate[fk.getJointParent(order_j)],
				localTransforms[order_j],
				localTranslate[order_j],
				globalTransforms[order_j],
				globalTranslate[order_j]
			);
	}

	// output handle positions.
	for (int i = 0; i < numIKJoints; i++)
	{
		// x, y, z
		handlePositions[i * 3] = globalTranslate[IKJointIDs[i]][0];
		handlePositions[i * 3 + 1] = globalTranslate[IKJointIDs[i]][1];
		handlePositions[i * 3 + 2] = globalTranslate[IKJointIDs[i]][2];
	}

}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  // Students should implement this.
  // fkinputdim, i<fkoutputdim for m n
  // Here, you should setup adol_c:
  //   Define adol_c inputs and outputs. 
  //   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
  //   This will later make it possible for you to compute the gradient of this function in IK::doIK
  //   (in other words, compute the "Jacobian matrix" J).
  // See ADOLCExample.cpp .

	// dimensions:
	int n = FKInputDim, m = FKOutputDim;

	trace_on(adolc_tagID);

	// input angles:

	vector<adouble> eu_angles(n);
	for (int i = 0; i < fk->getNumJoints(); i++)
	{
		eu_angles[i * 3] <<= 0.0;
		eu_angles[i * 3 + 1] <<= 0.0;
		eu_angles[i * 3 + 2] <<= 0.0;
	}

	// computation of handle_pos:

	vector<adouble> handle_pos(m);
	forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, eu_angles, handle_pos);

	// output:

	vector<double> output(m);

	for (int i = 0; i < m; i++)
	{
		handle_pos[i] >>= output[i];
	}

	trace_off();
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
	// You may find the following helpful:
	int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!

	// Students should implement this.
	// Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
	// Specifically, use ::function, and ::jacobian .
	// See ADOLCExample.cpp .
	//
	// Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
	// Note that at entry, "jointEulerAngles" contains the input Euler angles. 
	// Upon exit, jointEulerAngles should contain the new Euler angles.

	vector<double> handle_pos(FKOutputDim);	// handle position

	::function(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), handle_pos.data());

	vector<double> jacobMatrix(FKInputDim * FKOutputDim);	// jacobian matrix

	vector<double *> jacobMatrixEachRow(FKOutputDim);

	for (int i = 0; i < FKOutputDim; i++)
	{
		jacobMatrixEachRow[i] = &jacobMatrix[i*FKInputDim];
	}

	::jacobian(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), jacobMatrixEachRow.data());

	double alpha = 0.001;

	// Eigen Values; Jacobian, Jacobian Transpose, and Identity Matrices.

	Eigen::MatrixXd Eigen_J(FKOutputDim, FKInputDim), Eigen_J_Transpose, ID_Matrix;

	for (int i = 0; i < FKOutputDim; i++)
	{
		for (int j = 0; j < FKInputDim; j++)
		{
			Eigen_J(i, j) = jacobMatrix[i * FKInputDim + j];
		}
	}

	Eigen_J_Transpose = Eigen_J.transpose();

	ID_Matrix = ID_Matrix.Identity(FKInputDim, FKInputDim);

	// delta b calculation:

	vector<double> delta_b(FKOutputDim);
	Eigen::VectorXd delta_b_eigen(FKOutputDim);

	for (int i = 0; i < FKOutputDim; i++)
	{
		delta_b[i] = targetHandlePositions->data()[i] - handle_pos.data()[i];
		delta_b_eigen[i] = delta_b[i];
	}
	
	// Solve A:
	Eigen::MatrixXd A_var(FKInputDim, FKInputDim);
	A_var = Eigen_J_Transpose * Eigen_J + alpha * ID_Matrix;

	// Then B:
	Eigen::VectorXd B_var(FKOutputDim);
	B_var = Eigen_J_Transpose * delta_b_eigen;

	// To get the result, delta-theta:
	Eigen::VectorXd result_eigen = A_var.ldlt().solve(B_var);
	
	vector<double> result(FKInputDim);

	for(int i = 0; i < FKInputDim; i++)
		result[i] = result_eigen[i];

	// Output jointEulerAngles + delta-theta:
	for (int i = 0; i < numJoints; i++)
	{
		jointEulerAngles[i][0] = jointEulerAngles[i][0] + result[i * 3];
		jointEulerAngles[i][1] = jointEulerAngles[i][1] + result[i * 3 + 1];
		jointEulerAngles[i][2] = jointEulerAngles[i][2] + result[i * 3 + 2];
	}
}

